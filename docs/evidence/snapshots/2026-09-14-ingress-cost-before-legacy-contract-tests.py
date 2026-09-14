"""V1-02C frozen public-path source and admission contracts.

These exercise existing phase/storage boundaries, not a proposed API.
Injection seams assert the interleaving is exercised on both old and new code.
No provider or GPU is started. The original v1 diagnostics are archived.
"""

import os
from pathlib import Path

import pytest

from orze.core.fs import locked_append
from orze.engine.orchestrator import Orze
from orze.idea_lake import IdeaLake


def _block(idea_id, seed, title="Proposal"):
    return f"## {idea_id}: {title}\n```yaml\nseed: {seed}\n```\n\n"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    import orze.extensions

    monkeypatch.setattr(orze.extensions, "get_extension", lambda name: None)
    results = tmp_path / "results"
    results.mkdir()
    ideas_file = tmp_path / "ideas.md"
    ideas_file.write_text("# Ideas\n\n", encoding="utf-8")
    cfg = {
        "ideas_file": str(ideas_file),
        "results_dir": str(results),
        "idea_lake_db": str(tmp_path / "ideas.db"),
        "_orze_dir": str(tmp_path / ".orze"),
        "_env_ORZE_RESULTS_DIR": str(results),
        "sweep": {},
    }
    instance = Orze.__new__(Orze)
    instance.cfg = cfg
    instance.results_dir = results
    instance.lake = IdeaLake(cfg["idea_lake_db"])
    instance.failure_counts = {}
    instance.active_roles = {}
    try:
        yield instance, cfg, ideas_file
    finally:
        instance.lake.close()


def test_successful_append_after_parse_is_not_erased_without_admission(engine, monkeypatch):
    instance, cfg, ideas_file = engine
    ideas_file.write_text("# Ideas\n\n" + _block("idea-a", 1), encoding="utf-8")
    original_insert = instance.lake.insert
    appended = []

    def insert_then_append(*args, **kwargs):
        result = original_insert(*args, **kwargs)
        if not appended:
            appended.append(locked_append(
                ideas_file, _block("idea-b", 2),
                instance.results_dir / ".ideas_md.lock",
            ))
        return result

    monkeypatch.setattr(instance.lake, "insert", insert_then_append)
    instance._sync_ideas(cfg)
    assert len(appended) == 1, "the interleaving callback must actually execute"
    assert type(appended[0]) is bool
    if appended[0]:
        assert ("idea-b" in instance.lake.get_all_ids()
                or "## idea-b:" in ideas_file.read_text(encoding="utf-8")), (
            "successfully appended B disappeared from both source and SQLite"
        )
    else:
        # Correct consumer locking can defer the producer; exercise the real
        # retry after the consumer has released its source critical section.
        assert locked_append(
            ideas_file, _block("idea-b", 2),
            instance.results_dir / ".ideas_md.lock",
        )
    instance._sync_ideas(cfg)
    assert {"idea-a", "idea-b"} <= instance.lake.get_all_ids()
    assert "## idea-b:" not in ideas_file.read_text(encoding="utf-8")


def test_producer_finalizer_rollback_cannot_leave_a_queued_task(engine):
    instance, cfg, ideas_file = engine
    original = ideas_file.read_bytes()
    callbacks = []

    def failing_finalizer():
        callbacks.append("entered")
        instance._sync_ideas(cfg)
        raise ValueError("producer_admission_failed")

    with pytest.raises(ValueError, match="producer_admission_failed"):
        locked_append(
            ideas_file, _block("idea-uncommitted", 3),
            instance.results_dir / ".ideas_md.lock",
            after_append=failing_finalizer,
        )
    assert callbacks == ["entered"]
    assert ideas_file.read_bytes() == original, "producer really rolled back its append"
    assert instance.lake.get("idea-uncommitted") is None, (
        "consumer admitted bytes while the producer still owned the append/finalizer lock"
    )


def test_stale_id_snapshot_cannot_replace_a_concurrent_winner(engine, monkeypatch):
    instance, cfg, ideas_file = engine
    ideas_file.write_text("# Ideas\n\n" + _block("idea-shared", 1), encoding="utf-8")
    original_get_ids = instance.lake.get_all_ids
    competing = IdeaLake(cfg["idea_lake_db"])
    winner = []

    def stale_ids_then_competing_admission():
        old_ids = original_get_ids()
        if not winner:
            competing.insert(
                "idea-shared", "Concurrent winner", "seed: 99\n", "winner raw",
                status="completed", eval_metrics={"score": 17},
                priority="critical", hypothesis="winner hypothesis",
                created_at="2026-01-01T00:00:00Z",
            )
            winner.append(competing.get("idea-shared"))
        return old_ids

    monkeypatch.setattr(instance.lake, "get_all_ids", stale_ids_then_competing_admission)
    try:
        instance._sync_ideas(cfg)
        assert len(winner) == 1, "second connection must commit after the old snapshot"
        assert instance.lake.get("idea-shared") == winner[0], (
            "stale snapshot must not replace another connection's same-ID metadata/config"
        )
        assert instance.lake.get_fsm_state("idea-shared") == "COMPLETE"
    finally:
        competing.close()


def test_ignored_insert_does_not_consume_source_or_create_orphan_state(engine):
    instance, cfg, ideas_file = engine
    original = "# Ideas\n\n" + _block("idea-rejected", 4)
    ideas_file.write_text(original, encoding="utf-8")
    instance.lake.conn.executescript("""
        CREATE TRIGGER refuse_ingress BEFORE INSERT ON ideas
        WHEN NEW.idea_id = 'idea-rejected'
        BEGIN SELECT RAISE(IGNORE); END;
    """)
    instance._sync_ideas(cfg)
    assert instance.lake.get("idea-rejected") is None
    assert ideas_file.read_text(encoding="utf-8") == original, (
        "zero-row insertion was falsely acknowledged by clearing the source"
    )
    assert instance.lake.conn.execute(
        "SELECT 1 FROM idea_state WHERE idea_id = ?", ("idea-rejected",)
    ).fetchone() is None, "failed admission must not leave an orphan QUEUED state"


def test_replay_after_source_write_failure_acknowledges_existing_exact_task(engine, monkeypatch):
    instance, cfg, ideas_file = engine
    original = "# Ideas\n\n" + _block("idea-replay", 5)
    ideas_file.write_text(original, encoding="utf-8")
    path_write = Path.write_text
    os_replace = os.replace
    failed = []

    def fail_first_source_write(path, *args, **kwargs):
        if path == ideas_file and not failed:
            failed.append("write_text")
            raise OSError("injected_source_publish_failure")
        return path_write(path, *args, **kwargs)

    def fail_first_source_replace(source, target, *args, **kwargs):
        if Path(target) == ideas_file and not failed:
            failed.append("replace")
            raise OSError("injected_source_publish_failure")
        return os_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_first_source_write)
    monkeypatch.setattr(os, "replace", fail_first_source_replace)
    instance._sync_ideas(cfg)
    assert len(failed) == 1, "one real source publication boundary must be exercised"
    assert ideas_file.read_text(encoding="utf-8") == original
    admitted = instance.lake.get("idea-replay")
    assert admitted is not None
    transitions = list(instance.lake.conn.execute("SELECT * FROM idea_transitions"))
    instance._sync_ideas(cfg)
    assert instance.lake.get("idea-replay") == admitted
    assert list(instance.lake.conn.execute("SELECT * FROM idea_transitions")) == transitions
    assert "## idea-replay:" not in ideas_file.read_text(encoding="utf-8"), (
        "exact committed proposal must remain eligible for source ACK on a retry"
    )


def test_malformed_unadmitted_sibling_is_not_erased_by_successful_admission(engine):
    instance, cfg, ideas_file = engine
    malformed = "## idea-bad: Needs correction\n```yaml\nseed: [\n```\n"
    ideas_file.write_text(
        "# Ideas\n\n" + _block("idea-good", 6) + malformed, encoding="utf-8",
    )
    instance._sync_ideas(cfg)
    assert instance.lake.get_all_ids() == {"idea-good"}
    assert malformed in ideas_file.read_text(encoding="utf-8"), (
        "parse failure is not a durable rejection receipt authorizing deletion"
    )


def test_sidecar_is_additive_and_not_consumed(engine, tmp_path):
    instance, cfg, ideas_file = engine
    ideas_file.write_text("# Ideas\n\n" + _block("idea-main", 7), encoding="utf-8")
    sidecar = tmp_path / "ideas.d" / "idea-sidecar.md"
    sidecar.parent.mkdir()
    content = _block("idea-sidecar", 8)
    sidecar.write_text(content, encoding="utf-8")
    instance._sync_ideas(cfg)
    assert instance.lake.get_all_ids() == {"idea-main", "idea-sidecar"}
    assert "## idea-main:" not in ideas_file.read_text(encoding="utf-8")
    assert sidecar.read_text(encoding="utf-8") == content
