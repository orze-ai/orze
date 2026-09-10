"""Persistence checks for the new scoped operational-history mechanism.

These are reproducible regression checks, not old-version behavioral-red claims.
Use actual SQLite connections, public promotion checks, and an SQLite abort
trigger; no mocked storage implementation, evaluator, or network is involved.
"""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from orze.engine import champion_guard, champion_history
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "report": {
            "primary_metric": "quality", "sort": "descending",
            "columns": [{"key": "quality", "source": "evaluation.json:quality"}],
        },
        "champion_guard": {
            "enabled": True, "min_history": 50, "history_size": 50,
            "z_threshold": 4.0,
        },
    }
    try:
        yield SimpleNamespace(results=results, lake=lake, cfg=cfg)
    finally:
        lake.close()


def _publish(p, idea_id, value):
    folder = p.results / idea_id
    folder.mkdir()
    metrics = {"status": "COMPLETED", "quality": 999}
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (folder / "evaluation.json").write_text(
        json.dumps({"quality": value}), encoding="utf-8")
    p.lake.insert(idea_id, idea_id, "seed: 7", "", status="completed",
                  eval_metrics=metrics)


def _check(p, idea_id, value):
    # Do not share p.lake's connection across worker threads. The public guard
    # opens its own real qualification/history connections using cfg's DB path.
    return champion_guard.check_promotion(p.results, idea_id, value, p.cfg)


def _history_rows(p):
    return [tuple(row) for row in p.lake.conn.execute(
        "SELECT * FROM champion_guard_history_v1 ORDER BY sequence")]


def test_concurrent_promotion_upserts_keep_unique_and_distinct_idea_rows(project):
    p = project
    for idea_id, value in (
        ("idea-same", 1.0), ("idea-other-a", 2.0), ("idea-other-b", 3.0),
    ):
        _publish(p, idea_id, value)

    def check(item):
        return _check(p, *item)

    with ThreadPoolExecutor(max_workers=2) as pool:
        same = list(pool.map(check, [("idea-same", 1.0), ("idea-same", 1.0)]))
        different = list(pool.map(
            check, [("idea-other-a", 2.0), ("idea-other-b", 3.0)]))

    assert all(allowed for allowed, _ in same + different), same + different
    scope = champion_history.objective_scope(p.cfg)
    rows = [tuple(row) for row in p.lake.conn.execute(
        "SELECT idea_id, COUNT(*), metric FROM champion_guard_history_v1 "
        "WHERE scope=? GROUP BY idea_id ORDER BY idea_id", (scope,))]
    assert rows == [
        ("idea-other-a", 1, 2.0), ("idea-other-b", 1, 3.0), ("idea-same", 1, 1.0),
    ]


def test_real_history_write_failure_rejects_promotion_without_claiming_a_save(project):
    p = project
    _publish(p, "idea-seed", 1.0)
    allowed, info = _check(p, "idea-seed", 1.0)
    assert allowed, info
    before = _history_rows(p)
    p.lake.conn.execute(
        "CREATE TRIGGER reject_history_write BEFORE INSERT "
        "ON champion_guard_history_v1 WHEN NEW.idea_id='idea-failed-write' "
        "BEGIN SELECT RAISE(ABORT, 'injected history write failure'); END")
    p.lake.conn.commit()
    _publish(p, "idea-failed-write", 4.0)

    allowed, info = _check(p, "idea-failed-write", 4.0)

    assert allowed is False
    assert info["blocked"] is True
    assert info["reason"] == "promotion_check_unavailable"
    assert info["verified"] == 4.0  # Qualification succeeded before storage failed.
    assert _history_rows(p) == before
    assert p.lake.get_fsm_state("idea-failed-write") == "COMPLETE"


def test_history_and_promotion_never_create_a_missing_authority_database(project):
    p = project
    _publish(p, "idea-seed", 1.0)
    allowed, seed_info = _check(p, "idea-seed", 1.0)
    assert allowed, seed_info
    missing = p.results.parent / "missing-history.db"
    cfg_without_authority = dict(p.cfg, idea_lake_db=str(missing))

    allowed, info = champion_guard.check_promotion(
        p.results, "idea-seed", 1.0, cfg_without_authority)

    assert allowed is False
    assert info["blocked"] is True
    assert info["reason"] == "authoritative_lifecycle_database_unavailable"
    with pytest.raises(sqlite3.OperationalError):
        champion_history.record_history(
            missing, champion_history.objective_scope(p.cfg), "idea-seed", 1.0,
            seed_info["evidence_identity"], 50)
    assert not missing.exists()
    assert list(missing.parent.glob(missing.name + "*")) == []
