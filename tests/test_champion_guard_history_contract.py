"""A promotion heuristic is optional and its samples are qualified idea IDs."""

import json
from unittest.mock import Mock

import pytest

from orze.core.benchmark_contract import prepare_benchmark_evaluation
from orze.engine import champion_guard
from orze.idea_lake import IdeaLake
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
)


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    cfg = {
        "_project_root": str(tmp_path),
        "_env_ORZE_RESULTS_DIR": str(results),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "report": {
            "primary_metric": "quality",
            "sort": "descending",
            "columns": [
                {"key": "quality", "source": "evaluation.json:quality"},
            ],
        },
        "champion_guard": {
            "enabled": True,
            "min_history": 4,
            "history_size": 50,
            "z_threshold": 4.0,
        },
    }
    try:
        yield results, lake, cfg
    finally:
        lake.close()


def _enable_benchmark(project):
    # Reuse the exact sealed-evaluator receipt fixture, not a mocked verifier.
    from test_benchmark_contract import _config

    results, _, cfg = project
    guard_cfg = cfg["champion_guard"]
    cfg.update(_config(results.parent))
    cfg["champion_guard"] = guard_cfg
    cfg["eval_output"] = "evaluation.json"
    cfg["report"].update(
        primary_metric="quality",
        sort="descending",
        columns=[
            {"key": key, "source": f"evaluation.json:{key}"}
            for key in ("quality", "metric_a", "metric_b")
        ],
    )


def _write_source(project, idea_id, value):
    results, _, _ = project
    (results / idea_id / "evaluation.json").write_text(
        json.dumps({
            "quality": value,
            "alternate_quality": value,
            "metric_a": value,
            "metric_b": value,
        }),
        encoding="utf-8",
    )


def _qualified_identity(project, idea_id, expected):
    results, lake, cfg = project
    completed, reason = authoritative_completed_idea_ids(lake.db_path)
    assert reason == "authoritative_lifecycle_loaded"
    _, _, value, reason, identity = (
        qualify_authoritative_report_evidence_with_identity(
            idea_id, results, cfg, completed,
        )
    )
    assert value == expected, reason
    assert identity is not None
    return identity


def _publish(project, idea_id, value, *, misleading_score=False):
    results, lake, cfg = project
    folder = results / idea_id
    folder.mkdir()
    raw = {"status": "COMPLETED", "quality": 999}
    if misleading_score:
        raw["score"] = 0
    (folder / "metrics.json").write_text(json.dumps(raw), encoding="utf-8")
    _write_source(project, idea_id, value)
    lake.insert(idea_id, idea_id, "{}", "", status="completed", eval_metrics=raw)
    if cfg["report"].get("benchmark_contract"):
        from test_benchmark_contract import _write_receipt

        env = prepare_benchmark_evaluation(folder, cfg)
        _write_receipt(folder, cfg, env["ORZE_BENCHMARK_EVALUATION_NONCE"])
    _qualified_identity(project, idea_id, value)
    return idea_id


def _check(project, idea_id, value, **kwargs):
    results, _, cfg = project
    return champion_guard.check_promotion(results, idea_id, value, cfg, **kwargs)


def _seed(project, values=(-2.0, -1.0, 0.0, 1.0)):
    for index, value in enumerate(values):
        idea_id = _publish(project, f"idea-history-{index}", value)
        allowed, info = _check(project, idea_id, value)
        assert allowed, info


@pytest.mark.parametrize("policy", ["absent", "explicitly_disabled"])
def test_optional_policy_is_off_and_legacy_bare_history_is_untouched(project, policy):
    results, _, cfg = project
    if policy == "absent":
        cfg.pop("champion_guard")
    else:
        cfg["champion_guard"]["enabled"] = False
    legacy = results / "_champion_history.json"
    legacy_bytes = b'{"metrics": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}\n'
    legacy.write_bytes(legacy_bytes)
    idea_id = _publish(project, "idea-candidate", 100.0)

    allowed, info = _check(project, idea_id, 100.0)

    assert allowed, info
    assert info["enabled"] is False
    assert info["verified"] == 100.0
    assert legacy.read_bytes() == legacy_bytes


@pytest.mark.parametrize("direction,candidate", [
    ("ascending", -100.0), ("descending", 100.0),
])
def test_explicit_hold_uses_exact_source_and_both_anomaly_directions(
    project, direction, candidate,
):
    _, _, cfg = project
    cfg["report"]["sort"] = direction
    _seed(project)
    idea_id = _publish(project, "idea-candidate", candidate, misleading_score=True)

    allowed, info = _check(project, idea_id, candidate)

    assert allowed is False
    assert info["blocked"] is True
    assert info["verified"] == candidate
    assert info["history_size"] == 4
    assert abs(info["z"]) > cfg["champion_guard"]["z_threshold"]


def test_warn_policy_reports_anomaly_without_holding_or_implicit_audit(
    project, monkeypatch,
):
    _, lake, cfg = project
    cfg["champion_guard"]["action"] = "warn"
    _seed(project)
    idea_id = _publish(project, "idea-candidate", 100.0)
    audit = Mock()
    monkeypatch.setattr(champion_guard, "create_audit_idea", audit)
    count_before = lake.conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0]

    allowed, info = _check(project, idea_id, 100.0)

    assert allowed, info
    assert info["blocked"] is False
    assert abs(info["z"]) > cfg["champion_guard"]["z_threshold"]
    audit.assert_not_called()
    assert lake.conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0] == count_before


@pytest.mark.parametrize("changed_contract", ["primary", "source", "benchmark"])
def test_changed_objective_source_or_benchmark_cannot_borrow_old_samples(
    project, changed_contract,
):
    _, _, cfg = project
    if changed_contract == "benchmark":
        _enable_benchmark(project)
    _seed(project)
    if changed_contract == "primary":
        cfg["report"]["primary_metric"] = "other_quality"
        cfg["report"]["columns"] = [
            {"key": "other_quality", "source": "evaluation.json:quality"},
        ]
    elif changed_contract == "source":
        cfg["report"]["columns"][0]["source"] = "evaluation.json:alternate_quality"
    else:
        # A genuinely new pinned dataset is a legal contract change. Merely
        # relabeling the same dataset is correctly rejected by the ledger.
        cfg["report"]["benchmark_contract"].update(
            revision="b" * 40, dataset_manifest_sha256="c" * 64,
        )
    idea_id = _publish(project, "idea-new-contract", 100.0)

    allowed, info = _check(project, idea_id, 100.0)

    assert allowed, info
    assert info["history_size"] == 0
    assert info["z"] is None


def test_rechecking_and_revising_one_idea_cannot_inflate_distinct_history(project):
    _, _, cfg = project
    cfg["champion_guard"]["min_history"] = 50
    idea_id = _publish(project, "idea-revised", 1.0)
    for value in (1.0, 1.0, 1.25, 1.5):
        _write_source(project, idea_id, value)
        _qualified_identity(project, idea_id, value)
        allowed, info = _check(project, idea_id, value)
        assert allowed, info
        assert info["history_size"] == 0  # Candidate ID is never its own prior sample.

    # A stored sample whose current primary changed without a fresh guard
    # observation must not remain a valid historical measurement.
    _write_source(project, idea_id, 999.0)
    probe = _publish(project, "idea-probe", 2.0)
    allowed, info = _check(project, probe, 2.0)
    assert allowed, info
    assert info["history_size"] == 0


def test_shared_exposure_ledger_change_is_not_an_extra_independent_sample(project):
    _enable_benchmark(project)
    idea_id = _publish(project, "idea-first", 0.0)
    assert _check(project, idea_id, 0.0)[0]
    before = _qualified_identity(project, idea_id, 0.0)

    # This valid evaluation advances the shared ledger but is not a guard
    # observation. Its bytes belong to the first idea's snapshot identity too.
    _publish(project, "idea-ledger-only", 1.0)
    after = _qualified_identity(project, idea_id, 0.0)
    assert after != before
    allowed, info = _check(project, idea_id, 0.0)
    assert allowed, info
    assert info["history_size"] == 0

    probe = _publish(project, "idea-probe", 2.0)
    allowed, info = _check(project, probe, 2.0)
    assert allowed, info
    assert info["history_size"] == 1  # Unchanged first idea remains a usable sample.


@pytest.mark.parametrize("history", [(-1.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)])
def test_insufficient_or_flat_history_is_not_evidence_of_an_anomaly(project, history):
    _seed(project, history)
    idea_id = _publish(project, "idea-candidate", 100.0)

    allowed, info = _check(project, idea_id, 100.0)

    assert allowed, info
    assert info["history_size"] == len(history)
    assert info["z"] is None
