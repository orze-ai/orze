"""Actual claim/reset/cleanup must honor the native task's persisted DB route."""
import json
import os
from pathlib import Path
import socket
import sqlite3

import pytest

from orze.core.execution_attempts import create_attempt, finish_attempt, hold_attempt, mark_running
from orze.engine import failure, scheduler
from orze.engine.attempt_effect_lock import attempt_effect_lock
from orze.engine.execution_catalog import bind_catalog
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.idea_lake import IdeaLake


@pytest.fixture
def case(tmp_path, monkeypatch):
    results = tmp_path / "results"
    folder = results / "idea-native-route"
    lake = IdeaLake(tmp_path / "custom-native.sqlite")
    lake.insert(folder.name, "Native catalog route", "{}", "", status="queued")
    with attempt_effect_lock(folder) as lease:
        bind_catalog(lake, folder, lease)
    monkeypatch.setattr(scheduler, "capture_process_identity", lambda pid: {"start_ticks": 13})
    monkeypatch.setattr(scheduler, "process_is_running", lambda *args: False)
    try:
        yield results, folder, lake
    finally:
        lake.close()


def _attempt(lake, idea, phase, state):
    lake.conn.execute("BEGIN IMMEDIATE")
    ref = create_attempt(lake.conn, idea, phase, f"{phase}-A", {})
    if state in {"RUNNING", "TERMINAL"}:
        mark_running(lake.conn, ref)
    if state == "TERMINAL":
        assert finish_attempt(lake.conn, ref, {"outcome": "failed"}) == "committed"
    if state == "IN_DOUBT":
        hold_attempt(lake.conn, ref, "test unconfirmed process")
    lake.conn.commit()


def _evidence(folder, *, legacy_claim=False):
    (folder / "metrics.json").write_text('{"status":"PARTIAL","step":3}')
    (folder / "train_output.log").write_text("preserved training output")
    if legacy_claim:
        (folder / "claim.json").write_text(json.dumps({
            "attempt_id": "training-legacy-A", "gpu": 0,
            "claimed_by": socket.gethostname(), "pid": 991991,
            "owner_start_ticks": 13,
        }))
    for name in ("claim.json", "train_output.log"):
        if (folder / name).exists():
            os.utime(folder / name, (1, 1))


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def test_no_lake_claim_cannot_pass_native_evaluation_launching_catalog(case):
    results, folder, lake = case
    _attempt(lake, folder.name, "evaluation", "LAUNCHING")
    before = _files(folder)
    assert scheduler.claim(folder.name, results, 0) is False
    assert _files(folder) == before
    assert lake.get_fsm_state(folder.name) == "QUEUED"


@pytest.mark.parametrize("legacy_claim", [False, True], ids=["no-claim", "unbound-claim"])
def test_no_lake_reset_preserves_native_open_evaluation_without_claim_db_binding(case, legacy_claim):
    _, folder, lake = case
    _attempt(lake, folder.name, "evaluation", "RUNNING")
    _evidence(folder, legacy_claim=legacy_claim)
    before = _files(folder)
    with pytest.raises(TerminationUnconfirmed):
        failure._reset_idea_for_retry(folder, release_claim=True)
    assert _files(folder) == before


def test_no_lake_cleanup_preserves_catalog_bound_open_evaluation(case):
    results, folder, lake = case
    _attempt(lake, folder.name, "evaluation", "IN_DOUBT")
    _evidence(folder, legacy_claim=True)
    before = _files(folder)
    assert scheduler.cleanup_orphans(results, 1) == 0
    assert _files(folder) == before


@pytest.mark.parametrize("damage", ["malformed", "missing-db", "corrupt-db", "symlink", "unreadable"])
def test_unavailable_native_route_never_downgrades_reset_or_creates_database(case, monkeypatch, damage):
    results, folder, _ = case
    _evidence(folder)
    declaration = folder / "_execution_catalog.json"
    missing = results.parent / "must-not-create.sqlite"
    if damage == "malformed":
        declaration.write_text("{")
    elif damage in {"missing-db", "corrupt-db"}:
        value = json.loads(declaration.read_text())
        value["database"] = str(missing)
        if damage == "corrupt-db":
            missing.write_bytes(b"not a SQLite database")
        declaration.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    elif damage == "symlink":
        external = results.parent / "catalog-copy.json"
        external.write_bytes(declaration.read_bytes())
        declaration.unlink()
        declaration.symlink_to(external)
    else:
        real_open = os.open

        def denied(path, *args, **kwargs):
            if Path(path) == declaration:
                raise PermissionError("synthetic unreadable native route")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", denied)
    before = _files(folder)
    with pytest.raises(TerminationUnconfirmed):
        failure._reset_idea_for_retry(folder, release_claim=True)
    assert _files(folder) == before
    assert missing.exists() is (damage == "corrupt-db")
    if damage == "corrupt-db":
        assert missing.read_bytes() == b"not a SQLite database"


@pytest.mark.parametrize("other_route", ["supplied-lake", "claim"])
def test_conflicting_existing_main_routes_cannot_choose_a_more_permissive_database(case, other_route):
    results, folder, lake = case
    _attempt(lake, folder.name, "evaluation", "RUNNING")
    _evidence(folder, legacy_claim=other_route == "claim")
    peer = IdeaLake(results.parent / "unrelated.sqlite")
    try:
        if other_route == "claim":
            claim = json.loads((folder / "claim.json").read_text())
            claim["lifecycle_db"] = str(Path(peer.db_path).absolute())
            (folder / "claim.json").write_text(json.dumps(claim))
        before = _files(folder)
        with pytest.raises(TerminationUnconfirmed):
            failure._reset_idea_for_retry(
                folder, release_claim=True, lake=peer if other_route == "supplied-lake" else None)
        assert _files(folder) == before
    finally:
        peer.close()


@pytest.mark.parametrize("history", ["closed-both-phases", "no-native-rows"])
def test_valid_closed_catalog_uses_readonly_existing_database_and_allows_legacy_claim(case, monkeypatch, history):
    results, folder, lake = case
    if history == "closed-both-phases":
        for phase in ("training", "evaluation"):
            _attempt(lake, folder.name, phase, "TERMINAL")
    connections = []
    real_connect = sqlite3.connect

    def connect(database, *args, **kwargs):
        connections.append((database, kwargs.get("uri")))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    assert scheduler.claim(folder.name, results, 0)
    claim = json.loads((folder / "claim.json").read_text())
    assert claim.get("lifecycle_db") == str(Path(lake.db_path).absolute())
    assert connections and all(str(path).endswith("?mode=ro") and uri is True
                               for path, uri in connections)
