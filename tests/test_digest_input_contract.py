"""Digest labels distinguish observed artifacts from qualified conclusions."""

import json
import re

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
)
from orze.research.context_builder import build_digest


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
            "primary_metric": "score",
            "sort": "descending",
            "columns": [{"key": "score"}],
        },
    }
    try:
        yield results, lake, cfg
    finally:
        lake.close()


def _artifact(project, status, *, lifecycle="queued"):
    results, lake, _ = project
    idea_id = "idea-observed"
    folder = results / idea_id
    folder.mkdir()
    metrics = {"status": status, "score": 0.5}
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    lake.insert(
        idea_id, idea_id, "{}", "", status=lifecycle, eval_metrics=metrics
    )
    return idea_id


def _failure_section(digest):
    match = re.search(
        r"^## [^\n]*[Ff]ail[^\n]*\n(.*?)(?=^## |\Z)",
        digest, re.MULTILINE | re.DOTALL,
    )
    return match.group(0) if match else ""


@pytest.mark.parametrize("status", ["IN_PROGRESS", "QUEUED"])
def test_nonterminal_artifact_is_not_reported_as_a_recent_failure(project, status):
    results, _, cfg = project
    idea_id = _artifact(project, status)

    digest = build_digest(results, cfg)

    assert idea_id not in _failure_section(digest)
    assert not re.search(r"\b[1-9]\d* recent failures\b", digest)


@pytest.mark.parametrize("status", ["FAILED", "ERROR", "PARTIAL"])
def test_failure_artifacts_are_retained_as_explicitly_unverified_observations(
    project, status
):
    results, lake, cfg = project
    idea_id = _artifact(project, status)
    # The current lifecycle remains queued. This artifact is useful diagnostic
    # context, but its stale status must not be presented as current authority.
    assert lake.get(idea_id)["status"] == "queued"
    assert lake.get_fsm_state(idea_id) == "QUEUED"

    digest = build_digest(results, cfg)

    section = _failure_section(digest)
    assert idea_id in section
    heading = section.splitlines()[0].lower()
    assert "artifact-observed" in heading
    assert "unverified" in heading
    assert "(0 qualified candidates," in digest


def test_missing_primary_does_not_guess_score_or_emit_an_empty_objective_label(
    project
):
    results, lake, cfg = project
    idea_id = _artifact(project, "COMPLETED", lifecycle="completed")
    cfg["report"].pop("primary_metric")
    completed, reason = authoritative_completed_idea_ids(lake.db_path)
    assert reason == "authoritative_lifecycle_loaded"
    # Existing lower-level compatibility defaults accept this local score;
    # the digest must still require an explicit objective for research steering.
    _, _, value, _, _ = qualify_authoritative_report_evidence_with_identity(
        idea_id, results, cfg, completed
    )
    assert value == 0.5

    digest = build_digest(results, cfg)

    assert "(0 qualified candidates," in digest
    assert "evidence_mode: unavailable" in digest
    assert idea_id not in digest
    assert not re.search(r"^primary_metric:\s+sort:", digest, re.MULTILINE)
    assert not re.search(r"^## Top-[^\n]* by\s*$", digest, re.MULTILINE)


@pytest.mark.parametrize("direction", ["sideways", None, True])
def test_invalid_sort_produces_unavailable_digest_instead_of_claiming_a_rank(
    project, direction
):
    results, _, cfg = project
    idea_id = _artifact(project, "COMPLETED", lifecycle="completed")
    cfg["report"]["sort"] = direction

    digest = build_digest(results, cfg)

    assert "(0 qualified candidates," in digest
    assert "evidence_mode: unavailable" in digest
    assert idea_id not in digest
