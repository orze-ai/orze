"""Research digest steering must use the shared authoritative evidence policy."""

import json
import os
import re

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
)
from orze.reporting.objective import objective_sort_key
from orze.research.context_builder import build_digest


@pytest.fixture
def project_factory(tmp_path):
    lakes = []

    def make(name="project", *, direction="ascending"):
        root = tmp_path / name
        results = root / "results"
        results.mkdir(parents=True)
        database = root / "authority" / "research.sqlite3"
        database.parent.mkdir()
        lake = IdeaLake(str(database))
        lakes.append(lake)
        cfg = {
            "_project_root": str(root),
            "results_dir": str(results),
            "idea_lake_db": "authority/research.sqlite3",
            "report": {
                "primary_metric": "score",
                "sort": direction,
                "columns": [
                    {"key": "score", "source": "evaluation.json:score"}
                ],
            },
        }
        return root, results, lake, cfg

    yield make
    for lake in lakes:
        lake.close()


def _publish(project, idea_id, score, *, secondary=None, status="completed",
             tainted=False, mtime=100, family="model"):
    _, results, lake, _ = project
    folder = results / idea_id
    folder.mkdir()
    # The cache/proxy disagrees with the exact configured source on purpose.
    metrics = {"status": "COMPLETED", "score": -999, "penalty": 999}
    if tainted:
        metrics["tainted_leakage"] = True
    metrics_path = folder / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    os.utime(metrics_path, (mtime, mtime))
    (folder / "evaluation.json").write_text(
        json.dumps({"score": score, "penalty": secondary}), encoding="utf-8"
    )
    (folder / "idea_config.yaml").write_text(
        f"{family}: private-config-value\n", encoding="utf-8"
    )
    lake.insert(idea_id, idea_id, "{}", "", status=status, eval_metrics=metrics)


def _oracle(project):
    _, results, lake, cfg = project
    completed, reason = authoritative_completed_idea_ids(lake.db_path)
    assert reason == "authoritative_lifecycle_loaded"
    scoped_cfg = {**cfg, "_env_ORZE_RESULTS_DIR": str(results.resolve())}
    observations = []
    for folder in results.iterdir():
        if not folder.is_dir():
            continue
        _, values, value, _, identity = (
            qualify_authoritative_report_evidence_with_identity(
                folder.name, results, scoped_cfg, completed
            )
        )
        if value is not None:
            assert identity
            observations.append((folder.name, value, values))
    observations.sort(key=lambda row: objective_sort_key(
        row[1], row[2], cfg["report"], row[0]
    ))
    return observations


def _section(digest, heading_prefix):
    match = re.search(
        rf"^## {re.escape(heading_prefix)}[^\n]*\n(.*?)(?=^## |\Z)",
        digest, re.MULTILINE | re.DOTALL,
    )
    return match.group(1) if match else ""


def _ids(text):
    return re.findall(r"\bidea-[a-z0-9-]+\b", text)


def _qualified_count(digest):
    match = re.search(r"\((\d+) qualified candidates,", digest)
    assert match, digest
    return int(match.group(1))


@pytest.mark.parametrize("rejection", ["fsm_conflict", "queued", "tainted"])
def test_rejected_evidence_cannot_steer_top_recent_deltas_or_family_counts(
    project_factory, rejection
):
    project = project_factory()
    _, results, lake, cfg = project
    _publish(project, "idea-good-older", 1.0, mtime=100)
    _publish(project, "idea-good-recent", 2.0, mtime=200)
    _publish(
        project, "idea-rejected", 0.1, mtime=300, family="optimizer",
        status="queued" if rejection == "queued" else "completed",
        tainted=rejection == "tainted",
    )
    if rejection == "fsm_conflict":
        lake.conn.execute(
            "UPDATE idea_state SET current_state='QUEUED' WHERE idea_id=?",
            ("idea-rejected",),
        )
        lake.conn.commit()
    oracle_ids = [row[0] for row in _oracle(project)]
    assert oracle_ids == ["idea-good-older", "idea-good-recent"]

    digest = build_digest(results, cfg)

    family_counts = {
        family: int(count) for count, family in re.findall(
            r"^\s+(\d+)\s+([a-z-]+)\s*$",
            _section(digest, "Approach-family counts"), re.MULTILINE,
        )
    }
    assert {
        "qualified": _qualified_count(digest),
        "top": _ids(_section(digest, "Top-")),
        "recent": set(_ids(_section(digest, "Last-3"))),
        "families": family_counts,
    } == {
        "qualified": 2,
        "top": oracle_ids,
        "recent": set(oracle_ids),
        "families": {"model-configured": 2},
    }


def test_missing_authority_does_not_create_database_or_claim_qualified_results(
    project_factory
):
    project = project_factory()
    root, results, _, cfg = project
    _publish(project, "idea-artifact-only", 0.5)
    cfg["idea_lake_db"] = "absent/authority.sqlite3"
    missing_db = root / cfg["idea_lake_db"]
    assert not missing_db.exists()

    digest = build_digest(results, cfg)

    assert not missing_db.exists()
    assert not missing_db.parent.exists()
    assert _qualified_count(digest) == 0
    assert _ids(_section(digest, "Top-")) == []
    assert _ids(_section(digest, "Last-3")) == []
    assert _section(digest, "Approach-family counts") == ""
    assert "evidence_mode: unavailable" in digest


@pytest.mark.parametrize("direction", ["ascending", "descending"])
def test_digest_exact_primary_secondary_and_stable_ties_match_core_oracle(
    project_factory, direction
):
    project = project_factory(direction=direction)
    _, results, _, cfg = project
    cfg["report"]["secondary_metric"] = "penalty"
    cfg["report"]["columns"].append(
        {"key": "penalty", "source": "evaluation.json:penalty"}
    )
    for index, (idea_id, score, secondary) in enumerate([
        ("idea-z-tied", 5, 0),
        ("idea-b-tied", 5, 0),
        ("idea-negative", 5, -2),
        ("idea-positive", 5, 2),
        ("idea-a-missing", 5, None),
        ("idea-other-primary", 4, -999 if direction == "descending" else 999),
    ]):
        _publish(project, idea_id, score, secondary=secondary, mtime=100 + index)
    expected = ([
        "idea-other-primary", "idea-negative", "idea-b-tied", "idea-z-tied",
        "idea-positive", "idea-a-missing",
    ] if direction == "ascending" else [
        "idea-positive", "idea-b-tied", "idea-z-tied", "idea-negative",
        "idea-a-missing", "idea-other-primary",
    ])
    assert [row[0] for row in _oracle(project)] == expected

    digest = build_digest(results, cfg)

    assert _ids(_section(digest, "Top-")) == expected
    assert "score=5.0" in _section(digest, "Top-")
    assert "score=-999" not in digest


@pytest.mark.parametrize("root_hint", ["_project_root", "_config_path"])
def test_relative_custom_authority_is_project_scoped_and_projects_do_not_mix(
    project_factory, monkeypatch, root_hint
):
    first = project_factory("first")
    second = project_factory("second")
    _publish(first, "idea-shared", 0.25)
    _publish(first, "idea-first-only", 0.5)
    _publish(second, "idea-shared", 99, status="queued")
    _publish(second, "idea-second-only", 0.75)
    for root, _, _, cfg in (first, second):
        if root_hint == "_config_path":
            cfg.pop("_project_root")
            cfg["_config_path"] = str(root / "orze.yaml")
    first_expected = [row[0] for row in _oracle(first)]
    second_expected = [row[0] for row in _oracle(second)]
    assert first_expected == ["idea-shared", "idea-first-only"]
    assert second_expected == ["idea-second-only"]
    # Opposing the caller's cwd catches accidental resolution against cwd and
    # repeated calls catch module-global authority/result caches crossing roots.
    monkeypatch.chdir(second[0])
    first_digest = build_digest(first[1], first[3])
    monkeypatch.chdir(first[0])
    second_digest = build_digest(second[1], second[3])
    first_again = build_digest(first[1], first[3])

    assert {
        "first": _ids(_section(first_digest, "Top-")),
        "second": _ids(_section(second_digest, "Top-")),
        "first_again": _ids(_section(first_again, "Top-")),
    } == {
        "first": first_expected,
        "second": second_expected,
        "first_again": first_expected,
    }
    assert "score=0.25" in first_digest
    assert "score=99" not in first_digest
