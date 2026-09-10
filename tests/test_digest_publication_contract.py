"""V1-01D1: publish the real digest without replacing operator notes.

Exercise run_retrospection, including its interval acknowledgement. Generation
and publication failures are injected only at build_digest/atomic_write; the
successful path still generates from real lifecycle rows and local artifacts.
"""

import json
from pathlib import Path

import pytest

from orze.core import fs
from orze.engine import retrospection
from orze.idea_lake import IdeaLake
from orze.research import context_builder


_OLD_DIGEST = "Previous complete automatic digest.\n"
_OPERATOR_NOTES = "Operator hypothesis; not qualified experimental evidence.\n"
_LEGACY_NOTES = "Professor's legacy strategic notes; preserve verbatim.\n"


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    knowledge = results / "knowledge"
    knowledge.mkdir(parents=True)
    canonical = knowledge / "research_digest.md"
    canonical.write_text(_OLD_DIGEST, encoding="utf-8")
    (knowledge / "retrospection.md").write_text(
        _OPERATOR_NOTES, encoding="utf-8")
    (results / "_retrospection.txt").write_text(
        _LEGACY_NOTES, encoding="utf-8")

    idea_id = "idea-qualified"
    idea_dir = results / idea_id
    idea_dir.mkdir()
    metrics = {"status": "COMPLETED", "score": 2.0}
    (idea_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8")
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert(idea_id, idea_id, "{}", "", status="completed",
                eval_metrics=metrics)
    cfg = {
        "_project_root": str(tmp_path),
        "_env_ORZE_RESULTS_DIR": str(results),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "report": {
            "primary_metric": "score",
            "sort": "descending",
            "columns": [{"key": "score"}],
        },
        "retrospection": {"enabled": True, "interval": 2},
    }
    try:
        yield results, cfg, canonical
    finally:
        lake.close()


def _note_bytes(results):
    return {
        name: (results / name).read_bytes()
        for name in ("knowledge/retrospection.md", "_retrospection.txt")
    }


def _install_atomic_write(monkeypatch, writer):
    # Support the normal imported binding and a local import of the same
    # existing fs helper; do not replace run_retrospection itself.
    monkeypatch.setattr(fs, "atomic_write", writer)
    monkeypatch.setattr(retrospection, "atomic_write", writer, raising=False)


@pytest.mark.parametrize("existing_results", [False, True])
def test_disabled_retrospection_performs_no_generation_or_writes(
        tmp_path, monkeypatch, existing_results):
    results = tmp_path / "results"
    if existing_results:
        (results / "knowledge").mkdir(parents=True)
        (results / "knowledge" / "research_digest.md").write_text(
            _OLD_DIGEST, encoding="utf-8")
        (results / "knowledge" / "retrospection.md").write_text(
            _OPERATOR_NOTES, encoding="utf-8")
        (results / "_retrospection.txt").write_text(
            _LEGACY_NOTES, encoding="utf-8")
    before = {str(p.relative_to(tmp_path)): p.read_bytes()
              for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden_generation(*args, **kwargs):
        pytest.fail("disabled retrospection generated a digest")

    monkeypatch.setattr(context_builder, "build_digest", forbidden_generation)
    cfg = {"retrospection": {"enabled": False, "interval": 2}}

    returned = retrospection.run_retrospection(results, cfg, 100, 2)

    assert returned == 2
    assert results.exists() is existing_results
    assert {str(p.relative_to(tmp_path)): p.read_bytes()
            for p in tmp_path.rglob("*") if p.is_file()} == before


def test_interval_not_reached_does_not_publish_or_touch_notes(project, monkeypatch):
    results, cfg, canonical = project
    before_notes = _note_bytes(results)

    def forbidden_generation(*args, **kwargs):
        pytest.fail("retrospection generated before the interval elapsed")

    monkeypatch.setattr(context_builder, "build_digest", forbidden_generation)

    returned = retrospection.run_retrospection(results, cfg, 3, 2)

    assert returned == 2
    assert canonical.read_text(encoding="utf-8") == _OLD_DIGEST
    assert _note_bytes(results) == before_notes


def test_current_digest_is_atomically_published_without_overwriting_notes(
        project, monkeypatch):
    results, cfg, canonical = project
    before_notes = _note_bytes(results)
    generated = context_builder.build_digest(results, cfg)
    real_atomic_write = fs.atomic_write
    publications = []

    def observe_publication(path, content):
        path = Path(path)
        if path == canonical:
            # Readers still see the entire old publication before the atomic
            # replacement. The helper itself performs its real fsync+rename.
            assert path.read_text(encoding="utf-8") == _OLD_DIGEST
            assert content == generated
            assert _note_bytes(results) == before_notes
            publications.append(path)
        return real_atomic_write(path, content)

    _install_atomic_write(monkeypatch, observe_publication)

    returned = retrospection.run_retrospection(results, cfg, 4, 2)

    assert returned == 4
    assert publications == [canonical]
    assert canonical.read_text(encoding="utf-8") == generated
    assert "idea-qualified" in generated
    assert "score=2.0" in generated
    assert _note_bytes(results) == before_notes
    assert list(canonical.parent.glob("*.tmp")) == []


def test_generation_failure_is_not_acknowledged_and_retries_next_tick(
        project, monkeypatch):
    results, cfg, canonical = project
    before_notes = _note_bytes(results)
    real_build_digest = context_builder.build_digest
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("injected digest generation failure")
        return real_build_digest(*args, **kwargs)

    monkeypatch.setattr(context_builder, "build_digest", fail_once)

    first_count = retrospection.run_retrospection(results, cfg, 4, 2)
    first_publication = canonical.read_text(encoding="utf-8")
    second_count = retrospection.run_retrospection(results, cfg, 4, first_count)

    assert (first_count, second_count, len(attempts)) == (2, 4, 2)
    assert first_publication == _OLD_DIGEST
    assert canonical.read_text(encoding="utf-8") == real_build_digest(results, cfg)
    assert _note_bytes(results) == before_notes


@pytest.mark.parametrize("failure", ["raise", "silent_no_write"])
def test_publication_failure_preserves_last_digest_and_retries_next_tick(
        project, monkeypatch, failure):
    results, cfg, canonical = project
    before_notes = _note_bytes(results)
    real_atomic_write = fs.atomic_write
    publication_attempts = []

    def fail_once(path, content):
        path = Path(path)
        if path == canonical:
            publication_attempts.append(path)
            if len(publication_attempts) == 1:
                if failure == "raise":
                    raise OSError("injected atomic publication failure")
                # atomic_write currently swallows ENOSPC in its write loop;
                # lack of an exception must not acknowledge an absent publish.
                return None
        return real_atomic_write(path, content)

    _install_atomic_write(monkeypatch, fail_once)

    first_count = retrospection.run_retrospection(results, cfg, 4, 2)
    first_publication = canonical.read_text(encoding="utf-8")
    first_notes = _note_bytes(results)
    second_count = retrospection.run_retrospection(results, cfg, 4, first_count)

    assert (first_count, second_count, len(publication_attempts)) == (2, 4, 2)
    assert first_publication == _OLD_DIGEST
    assert first_notes == before_notes
    assert canonical.read_text(encoding="utf-8") == context_builder.build_digest(
        results, cfg)
    assert _note_bytes(results) == before_notes
    assert list(canonical.parent.glob("*.tmp")) == []
