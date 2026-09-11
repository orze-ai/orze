"""Opt-in SQL metadata watch mechanisms, not process/file closure evidence."""
import copy
from dataclasses import asdict
import json

import pytest

from orze.core.execution_attempts import AttemptAuthorityError, AttemptRef, create_attempt, mark_running
from orze.core.research_artifacts import artifacts_for_attempt, register_artifacts
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.execution_authority import execution_transaction
from orze.idea_lake import IdeaLake


@pytest.fixture
def catalog(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    folder = tmp_path / "results" / "idea-watch"
    folder.mkdir(parents=True)
    bound = {"contract": {"version": 1, "outputs": {"value": {"path": "value", "max_bytes": 5}}},
             "root": str(tmp_path / "artifacts"), "scope": str(folder.parent), "spec_fingerprint": "a" * 64}
    with execution_transaction(lake, folder) as tx:
        ref = create_attempt(tx.conn, folder.name, "metadata-test", "attempt-watch", {"artifact_publication": bound})
        mark_running(tx.conn, ref)
        record = {"schema": 1, "artifact_id": "artifact-watch", "producer": asdict(ref),
                  "logical_name": "value", "path": str(tmp_path / "artifacts" / "artifact-watch" / "content"),
                  "content_sha256": "b" * 64, "size_bytes": 3, "scope": bound["scope"],
                  "spec_fingerprint": bound["spec_fingerprint"]}
        register_artifacts(tx.conn, ref, [record])
    yield lake, folder, ref, record
    lake.close()


def test_artifact_watch_captures_expected_metadata_not_mutable_caller_list(catalog):
    lake, folder, ref, record = catalog
    supplied = [copy.deepcopy(record)]
    with execution_transaction(lake, folder) as tx:
        tx.watch_artifacts(ref, supplied)
        supplied[0]["content_sha256"] = "c" * 64
    assert artifacts_for_attempt(lake.conn, ref) == [record]


@pytest.mark.parametrize("case", ["foreign_ref", "too_many", "foreign_producer", "duplicate", "record_size"])
def test_artifact_watch_bounds_and_exact_reference(catalog, case):
    lake, folder, ref, record = catalog
    expected = [copy.deepcopy(record)]
    watched = ref
    if case == "foreign_ref":
        watched = AttemptRef("other-task", ref.phase, ref.attempt_id, ref.generation)
    elif case == "too_many":
        expected *= 33
    elif case == "foreign_producer":
        expected[0]["producer"]["attempt_id"] = "other-attempt"
    elif case == "duplicate":
        expected *= 2
    else:
        expected[0]["path"] = "/" + "x" * 17000
    with execution_transaction(lake, folder) as tx:
        with pytest.raises((AttemptAuthorityError, ValueError)):
            tx.watch_artifacts(watched, expected)
    assert artifacts_for_attempt(lake.conn, ref) == [record]


def test_artifact_watch_rechecks_after_actual_commit_before_returning_success(catalog):
    lake, folder, ref, record = catalog
    original = lake.conn
    changed = {**record, "content_sha256": "c" * 64}
    calls = []

    class AfterCommit:
        def __getattr__(self, key):
            return getattr(original, key)

        def commit(self):
            original.commit()
            original.execute("UPDATE research_artifacts SET record_json=? WHERE artifact_id=?",
                (json.dumps(changed, sort_keys=True, ensure_ascii=False, separators=(",", ":")), record["artifact_id"]))
            original.commit()
            calls.append("committed_artifact_rewrite")
            assert not original.in_transaction

    lake.conn = AfterCommit()
    try:
        with pytest.raises(AttemptEffectInDoubt):
            with execution_transaction(lake, folder) as tx:
                tx.watch_artifacts(ref, [record])
    finally:
        lake.conn = original
    assert calls == ["committed_artifact_rewrite"]
    assert artifacts_for_attempt(lake.conn, ref) == [changed]
