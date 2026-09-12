"""An explicit honesty declaration is a veto, not proof of scientific validity.

These contracts exercise real local artifacts and authoritative IdeaLake state.
They deliberately use an arbitrary declared source and a legitimate zero value.
"""

import json
import os
import re

import pytest

from orze.engine.rebuild_state import rebuild_state_file
from orze.idea_lake import IdeaLake
from orze.reporting import evidence, leaderboard
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence,
    qualify_local_report_evidence,
)
from orze.research.context_builder import build_digest


_MISSING = object()


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "idea_lake_db": lake.db_path,
        "report": {
            "primary_metric": "score",
            "sort": "ascending",
            "columns": [
                {"key": "score", "source": "evaluation.json:measurement.score"},
            ],
        },
        "metric_validation": {
            "min_value": {"score": -10},
            "max_value": {"score": 10},
        },
    }
    yield results, lake, cfg
    lake.close()


def _publish(project, idea_id, score, *, honest=_MISSING, mtime=100):
    results, lake, _ = project
    folder = results / idea_id
    folder.mkdir()
    metrics = {"status": "COMPLETED", "score": 99}
    if honest is not _MISSING:
        metrics["honest"] = honest
    metrics_path = folder / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    os.utime(metrics_path, (mtime, mtime))
    (folder / "evaluation.json").write_text(
        json.dumps({"measurement": {"score": score}}), encoding="utf-8"
    )
    lake.insert(
        idea_id, idea_id, "{}", "", status="completed",
        eval_metrics={"status": "COMPLETED", "score": -999},
    )
    return folder


def _top_ids(digest):
    section = re.search(
        r"^## Top-[^\n]*\n(.*?)(?=^## |\Z)",
        digest, re.MULTILINE | re.DOTALL,
    )
    assert section, digest
    return re.findall(r"\bidea-[a-z0-9-]+\b", section.group(1))


@pytest.mark.parametrize(
    "declaration", [False, 0, 1, "false", "true", None, [], {}],
    ids=["false", "integer-zero", "integer-one", "string-false",
         "string-true", "null", "array", "object"],
)
def test_local_qualifier_rejects_explicit_false_or_nonboolean_honesty(
    project, declaration
):
    _, _, cfg = project
    folder = _publish(project, "idea-rejected", 0, honest=declaration)

    metrics, values, primary, reason = qualify_local_report_evidence(folder, cfg)

    assert metrics["honest"] == declaration
    assert values["score"] == 0
    assert primary is None, "An explicit dishonest/malformed declaration must veto ranking"
    assert reason != "local_evidence_verified"


@pytest.mark.parametrize("consumer", ["report", "rebuild", "digest"])
@pytest.mark.parametrize(
    "declaration", [False, 0, "false"], ids=["false", "integer-zero", "string-false"]
)
def test_public_consumers_exclude_a_better_explicitly_rejected_candidate(
    project, consumer, declaration
):
    results, lake, cfg = project
    _publish(project, "idea-baseline", 1, mtime=100)
    _publish(project, "idea-rejected", 0, honest=declaration, mtime=200)

    if consumer == "report":
        rows = leaderboard.update_report(results, {}, cfg, lake=lake)
        assert [row["id"] for row in rows] == ["idea-baseline"]
        assert rows[0]["primary_val"] == 1
    elif consumer == "rebuild":
        summary = rebuild_state_file(results, cfg, lake=lake)
        assert summary["best_idea_id"] == "idea-baseline"
        assert summary["completions_since_best"] == 0
    else:
        digest = build_digest(results, cfg)
        assert _top_ids(digest) == ["idea-baseline"]
        assert "idea-rejected" not in digest


@pytest.mark.parametrize("declaration", [_MISSING, True], ids=["absent", "true"])
def test_absent_or_true_declaration_preserves_qualified_source_zero_without_proof(
    project, declaration
):
    results, lake, cfg = project
    folder = _publish(project, "idea-zero", 0, honest=declaration)

    metrics, values, primary, reason = qualify_local_report_evidence(folder, cfg)

    assert (values["score"], primary, reason) == (0, 0.0, "local_evidence_verified")
    if declaration is _MISSING:
        assert "honest" not in metrics
    rows = leaderboard.update_report(results, {}, cfg, lake=lake)
    assert [(row["id"], row["primary_val"]) for row in rows] == [("idea-zero", 0.0)]
    assert rebuild_state_file(results, cfg, lake=lake)["best_idea_id"] == "idea-zero"
    digest = build_digest(results, cfg)
    assert _top_ids(digest) == ["idea-zero"]
    assert "identity is not cryptographically proven" in digest


@pytest.mark.parametrize(
    "rejection", ["missing-source", "range", "coverage", "nonfinite", "taint", "fsm"]
)
def test_true_declaration_does_not_override_other_authoritative_requirements(
    project, rejection
):
    results, lake, cfg = project
    folder = _publish(project, "idea-rejected", 0, honest=True)
    if rejection == "missing-source":
        cfg["report"]["columns"][0]["source"] = "absent.json:measurement.score"
    elif rejection == "range":
        cfg["metric_validation"]["min_value"]["score"] = 1
    elif rejection == "coverage":
        cfg["report"]["columns"].append(
            {"key": "replicate", "source": "evaluation.json:measurement.replicate"}
        )
        cfg["report"]["min_datasets"] = 2
        cfg["report"]["dataset_keys"] = ["score", "replicate"]
    elif rejection == "nonfinite":
        (folder / "evaluation.json").write_text(
            json.dumps({"measurement": {"score": float("inf")}}), encoding="utf-8"
        )
    elif rejection == "taint":
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        metrics["tainted_leakage"] = True
        (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    else:
        lake.conn.execute(
            "UPDATE idea_state SET current_state='QUEUED' WHERE idea_id=?",
            ("idea-rejected",),
        )
        lake.conn.commit()

    completed, reason = authoritative_completed_idea_ids(lake.db_path)
    assert reason == "authoritative_lifecycle_loaded"
    qualified = qualify_authoritative_report_evidence(
        "idea-rejected", results, cfg, completed
    )
    assert qualified[2] is None
    assert rebuild_state_file(results, cfg, lake=lake)["best_idea_id"] is None
    assert "idea-rejected" not in build_digest(results, cfg)


def test_report_requalifies_old_accepted_cache_after_honesty_policy_change(
    project, monkeypatch
):
    results, lake, cfg = project
    _publish(project, "idea-baseline", 1)
    folder = _publish(project, "idea-rejected", 0, honest=False, mtime=200)
    original_qualifier = evidence.qualify_local_report_evidence

    def pre_honesty_policy(idea_dir, full_cfg, **kwargs):
        metrics, values, primary, reason = original_qualifier(
            idea_dir, full_cfg, **kwargs)
        if metrics.get("honest") is False:
            # Simulate only the previous policy's acceptance of this otherwise
            # valid fixture. Real update_report writes its own cache and hashes.
            assert values["score"] == 0
            return metrics, values, 0.0, "local_evidence_verified"
        return metrics, values, primary, reason

    with monkeypatch.context() as legacy:
        legacy.setattr(leaderboard, "_RESULT_CACHE_SCHEMA_VERSION", 5)
        legacy.setattr(evidence, "qualify_local_report_evidence", pre_honesty_policy)
        old_rows = leaderboard.update_report(results, {}, cfg, lake=lake)
        assert old_rows[0]["id"] == "idea-rejected"
    cache = json.loads((results / "_results_cache.json").read_text(encoding="utf-8"))
    assert cache["idea-rejected"]["cache_schema_version"] == 5
    assert cache["idea-rejected"]["row"]["evidence_qualified"] is True
    artifact_snapshot = {
        path.name: (path.read_bytes(), path.stat())
        for path in (folder / "metrics.json", folder / "evaluation.json")
    }

    rows = leaderboard.update_report(results, {}, cfg, lake=lake)

    assert [row["id"] for row in rows] == ["idea-baseline"], (
        "Unchanged artifact timestamps/content cannot preserve a qualification "
        "that the current canonical policy rejects"
    )
    assert {
        path.name: (path.read_bytes(), path.stat())
        for path in (folder / "metrics.json", folder / "evaluation.json")
    } == artifact_snapshot
