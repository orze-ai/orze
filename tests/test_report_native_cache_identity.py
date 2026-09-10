"""Native report caches cannot grant authority or invent observation identity.

All cases use a real IdeaLake and update_report. Files and lifecycle are real;
the sole fault injection changes a real source at its read boundary. Stable
identity is per row, not a promise of one atomic snapshot of the whole report.
The final test separately specifies the new evidence_identity output field;
its absence on the baseline is not an existing-behavior regression.
"""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    evidence_content_sha256,
    qualify_authoritative_report_evidence_with_identity,
    report_evidence_paths,
)
from orze.reporting.leaderboard import update_report


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "authority.db"))
    cfg = {
        "_project_root": str(tmp_path),
        "_orze_dir": str(tmp_path / ".orze"),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "eval_output": "observation.json",
        "report": {
            "title": "Native current observations",
            "primary_metric": "quality",
            "sort": "descending",
            "columns": [
                {"key": "quality", "source": "observation.json:quality"},
            ],
        },
    }
    p = SimpleNamespace(results=results, lake=lake, cfg=cfg, ideas={})
    try:
        yield p
    finally:
        lake.close()


def _publish(p, idea_id="idea-native", quality=1.0, **extra_metrics):
    folder = p.results / idea_id
    folder.mkdir()
    metrics = {"status": "COMPLETED", "quality": 999, **extra_metrics}
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (folder / "observation.json").write_text(
        json.dumps({"quality": quality}), encoding="utf-8")
    p.lake.insert(idea_id, idea_id, "seed: 19", "", status="completed",
                  eval_metrics=metrics)
    p.ideas[idea_id] = {"title": idea_id, "config": {"seed": 19}}
    return folder


def _report(p):
    return update_report(p.results, p.ideas, p.cfg, lake=p.lake)


def _qualify(p, idea_id="idea-native"):
    completed, reason = authoritative_completed_idea_ids(p.lake.db_path)
    assert reason == "authoritative_lifecycle_loaded"
    scoped_cfg = dict(p.cfg, _env_ORZE_RESULTS_DIR=str(p.results.resolve()))
    return qualify_authoritative_report_evidence_with_identity(
        idea_id, p.results, scoped_cfg, completed)


def _published_top(p):
    return json.loads((p.results / "_leaderboard.json").read_text(
        encoding="utf-8"))["top"]


def _assert_unranked(p, rows, idea_id="idea-native"):
    assert idea_id not in {row["id"] for row in rows}
    assert idea_id not in {row["idea_id"] for row in _published_top(p)}
    # Audit sections may retain the ID, but the actual rank table may not.
    text = (p.results / "report.md").read_text(encoding="utf-8")
    assert not any(line.startswith("| ") and f"| {idea_id} |" in line
                   for line in text.splitlines())


def test_warm_native_cache_cannot_bypass_new_clean_access_policy(project):
    p = project
    folder = _publish(p)
    (folder / "_access_log.tsv").write_text(
        "WATCH\t/private/eval\t/private/eval/sample.arrow\n", encoding="utf-8")
    assert len(_report(p)) == 1  # Policy was not yet requested.
    assert (p.results / "_results_cache.json").is_file()

    p.cfg["managed_run"] = {"require_clean_training_access_log": True}
    _, _, value, reason, _ = _qualify(p)
    assert value is None
    assert reason == "training_access_log_not_clean"

    _assert_unranked(p, _report(p))


@pytest.mark.parametrize("changed_log", [
    pytest.param("WATCH\t/private/eval\t/private/eval/sample.arrow\n", id="watch"),
    pytest.param("FORBIDDEN\t/private/eval\t/private/eval/sample.arrow\n",
                 id="forbidden"),
    pytest.param("unparseable access evidence\n", id="malformed"),
])
def test_warm_native_cache_requalifies_changed_access_log(project, changed_log):
    p = project
    p.cfg["managed_run"] = {"require_clean_training_access_log": True}
    folder = _publish(p)
    access_log = folder / "_access_log.tsv"
    access_log.write_text("", encoding="utf-8")
    assert len(_report(p)) == 1
    before_identity = _qualify(p)[4]

    access_log.write_text(changed_log, encoding="utf-8")
    _, _, value, reason, after_identity = _qualify(p)
    assert value is None
    assert reason == "training_access_log_not_clean"
    assert before_identity != after_identity

    _assert_unranked(p, _report(p))


@pytest.mark.parametrize("replacement,accepted", [(2.0, True), (9.0, False)])
def test_native_same_size_backdated_source_is_requalified(project, replacement, accepted):
    p = project
    p.cfg["metric_validation"] = {"max_value": {"quality": 5}}
    source = _publish(p) / "observation.json"
    assert _report(p)[0]["primary_val"] == 1.0
    before = source.stat()
    source.write_text(json.dumps({"quality": replacement}), encoding="utf-8")
    assert source.stat().st_size == before.st_size
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_mtime_ns == before.st_mtime_ns

    rows = _report(p)
    _, _, value, _, _ = _qualify(p)

    if accepted:
        assert value == replacement
        assert rows[0]["primary_val"] == replacement
        assert _published_top(p)[0]["metric_value"] == replacement
    else:
        assert value is None
        _assert_unranked(p, rows)


@pytest.mark.parametrize("honest", [True, False])
def test_rehashed_cache_row_cannot_invent_score_or_qualification(project, honest):
    p = project
    _publish(p, honest=honest)
    initial = _report(p)
    assert bool(initial) is honest
    cache_path = p.results / "_results_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    row = cache["idea-native"]["row"]
    row.update(primary_val=777.0, values={"quality": 777.0},
               metrics={"status": "COMPLETED", "quality": 777.0, "honest": True},
               status="COMPLETED", lifecycle_completed=True,
               evidence_qualified=True, evidence_reason="local_evidence_verified",
               benchmark_contract_ok=True)
    # An unkeyed row checksum detects corruption, not adversarial cache input.
    # Recompute it exactly while leaving actual evidence and metadata intact.
    cache["idea-native"]["row_hash"] = hashlib.sha256(json.dumps(
        row, sort_keys=True, default=str, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    rows = _report(p)
    _, _, value, reason, _ = _qualify(p)

    if honest:
        assert value == 1.0
        assert rows[0]["primary_val"] == 1.0
        assert rows[0]["values"]["quality"] == 1.0
        assert _published_top(p)[0]["metric_value"] == 1.0
    else:
        assert value is None
        assert reason == "local_evidence_declared_non_honest"
        _assert_unranked(p, rows)


def test_source_change_during_read_excludes_only_unstable_native_row(
    project, monkeypatch,
):
    p = project
    changing = _publish(p, quality=1.0) / "observation.json"
    _publish(p, idea_id="idea-stable", quality=2.0)
    assert len(_report(p)) == 2
    # Invalidate the previous cache before injecting a *second* mutation at
    # the actual source read; do not mock away the shared qualifier boundary.
    changing.write_text(json.dumps({"quality": 3.0}), encoding="utf-8")
    read_text = Path.read_text
    mutated = False

    def replace_at_read(path, *args, **kwargs):
        nonlocal mutated
        if path == changing and not mutated:
            mutated = True
            changing.write_text(json.dumps({"quality": 9.0}), encoding="utf-8")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", replace_at_read)

    rows = _report(p)

    assert mutated, "fault injection must reach the real observation read"
    _assert_unranked(p, rows)
    assert [row["id"] for row in rows] == ["idea-stable"]
    assert _published_top(p)[0]["metric_value"] == 2.0
    # A later stable read may accept the new observation. This test promises
    # fail-closed per-row reconciliation, not an immutable global snapshot.
    assert _qualify(p)[2] == 9.0


@pytest.mark.parametrize("warm", [False, True])
def test_native_success_row_publishes_shared_evidence_identity(project, warm):
    p = project
    p.cfg["managed_run"] = {"require_clean_training_access_log": True}
    folder = _publish(p, quality=0.0)
    (folder / "_access_log.tsv").write_text("", encoding="utf-8")
    if warm:
        assert len(_report(p)) == 1

    rows = _report(p)
    _, values, value, reason, identity = _qualify(p)

    assert reason == "authoritative_local_evidence_verified"
    assert rows[0]["primary_val"] == value == 0.0
    assert rows[0]["values"] == values
    assert identity == evidence_content_sha256(
        report_evidence_paths("idea-native", p.results, p.cfg))
    assert rows[0].get("evidence_identity") == identity, (
        "NEW_FIELD_CONTRACT: native rows must publish the shared stable "
        "evidence_identity; its baseline absence is not a behavioral red")
