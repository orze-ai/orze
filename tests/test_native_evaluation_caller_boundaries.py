"""Real evaluator consumers with external process/GPU and I/O seams only.

Tests reuse the unchanged public-workflow fixture, not mocked attempt checks,
qualification, lifecycle, retry, or receipt APIs. New draft behavior boundaries
are distinguished from old-release tests in the evidence ledger.
"""
from copy import copy
from pathlib import Path

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.evaluation_retry import request_evaluation_retry
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.idea_lake import IdeaLake

from test_stale_evaluation_completion import (
    project, _prepare, _launch, _exit, _files, _lifecycle,
)


@pytest.mark.parametrize("retained_token", [True, False], ids=["native-token", "tokenless-restored"])
def test_omitting_lake_never_downgrades_native_completion_to_legacy(project, retained_token):
    p = project
    folder = _prepare(p, "idea-no-lake")
    ep = _launch(p, folder.name)
    _exit(p, ep, 1)
    callback = copy(ep)
    if not retained_token:
        callback.attempt_ref = None
    before_files, before_state = _files(folder), _lifecycle(p.lake)
    active = {0: callback}

    with pytest.raises(TerminationUnconfirmed):
        evaluator.check_active_evals(active, p.results, p.cfg)

    assert active.get(0) is callback
    assert _files(folder) == before_files
    assert _lifecycle(p.lake) == before_state


def test_native_catalog_rejects_tokenless_handle_before_any_terminal_effect(project):
    p = project
    folder = _prepare(p, "idea-no-token")
    ep = _launch(p, folder.name)
    _exit(p, ep, 1)
    ep.attempt_ref = None
    before_files, before_state = _files(folder), _lifecycle(p.lake)
    active = {0: ep}

    with pytest.raises(TerminationUnconfirmed):
        evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)

    assert active.get(0) is ep
    assert _files(folder) == before_files
    assert _lifecycle(p.lake) == before_state


def test_popen_observes_committed_launch_intent_without_holding_effect_guard(project):
    p = project
    folder = _prepare(p, "idea-before-popen")
    original = p.popen.side_effect
    observed = []

    def observe_before_popen(*args, **kwargs):
        peer = IdeaLake(p.lake.db_path)
        try:
            row = current_attempt(peer.conn, folder.name, "evaluation")
            observed.append(row)
            assert row["state"] == "LAUNCHING"
            assert peer.get_stage_state(folder.name, "evaluation") == "IN_PROGRESS"
            assert not (folder / "_attempt_effect.lock").exists()
            assert not (folder / "_compute_receipts" / row["attempt_id"] / "start.json").exists()
        finally:
            peer.close()
        return original(*args, **kwargs)

    p.popen.side_effect = observe_before_popen
    ep = _launch(p, folder.name)
    assert len(observed) == 1
    assert observed[0]["attempt_id"] == ep.attempt_id
    assert current_attempt(p.lake.conn, folder.name, "evaluation")["state"] == "RUNNING"


def test_created_process_registration_write_rejection_stops_and_holds(project):
    p = project
    folder = _prepare(p, "idea-register-fault")
    from orze.core.execution_attempts import ensure_schema
    p.lake.conn.execute("BEGIN IMMEDIATE")
    ensure_schema(p.lake.conn)
    p.lake.conn.execute(
        "CREATE TRIGGER ignore_running BEFORE UPDATE ON execution_attempts "
        "WHEN NEW.state='RUNNING' BEGIN SELECT RAISE(IGNORE); END")
    p.lake.conn.commit()

    def confirmed_stop(process, *args, **kwargs):
        process.returncode = -15
        return True

    p.reaper.side_effect = confirmed_stop
    with pytest.raises(TerminationUnconfirmed):
        evaluator.launch_eval(folder.name, 0, p.results, p.cfg, lake=p.lake)

    p.popen.assert_called_once()
    p.reaper.assert_called_once()
    assert p.processes[0].returncode == -15
    row = current_attempt(p.lake.conn, folder.name, "evaluation")
    assert row["state"] in {"LAUNCHING", "IN_DOUBT"}
    assert p.lake.get_stage_state(folder.name, "evaluation") == "IN_PROGRESS"
    assert not (folder / "_compute_receipts" / row["attempt_id"] / "terminal.json").exists()
    with pytest.raises(TerminationUnconfirmed):
        evaluator.launch_eval(folder.name, 0, p.results, p.cfg, lake=p.lake)
    p.popen.assert_called_once()


def test_validation_read_can_lose_to_a_real_peer_retry_without_touching_new_attempt(project, monkeypatch):
    p = project
    folder = _prepare(p, "idea-validation-peer")
    old = _launch(p, folder.name)
    _exit(p, old, 0, valid=False)
    peer = IdeaLake(p.lake.db_path)
    actual_validate = evaluator.validate_evaluation_result
    active = {0: old}
    observations = []

    def validate_then_peer(*args, **kwargs):
        result = actual_validate(*args, **kwargs)
        if observations:
            return result
        observations.append("interleaving")
        assert evaluator.check_active_evals({0: old}, p.results, p.cfg, lake=peer) == [(folder.name, 0)]
        assert request_evaluation_retry(folder.name, p.results, p.cfg, peer)["status"] == "evaluation_retry_pending"
        current = evaluator.launch_eval(folder.name, 0, p.results, p.cfg, lake=peer)
        assert current is not None
        active[0] = current
        observations.append((current, _files(folder), _lifecycle(peer)))
        return result

    monkeypatch.setattr(evaluator, "validate_evaluation_result", validate_then_peer)
    try:
        try:
            finished = evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)
        except TerminationUnconfirmed:
            finished = []
        assert finished == []
        current, files, state = observations[1]
        assert active.get(0) is current
        assert _files(folder) == files
        assert _lifecycle(peer) == state
        assert p.lake.get_stage_state(folder.name, "evaluation") == "IN_PROGRESS"
        assert current.attempt_id != old.attempt_id
    finally:
        peer.close()


def test_actual_marker_write_keeps_peer_completion_outside_owned_effect_boundary(project, monkeypatch):
    p = project
    folder = _prepare(p, "idea-write-peer")
    ep = _launch(p, folder.name)
    ep.process.returncode = 1  # No evaluator output: the real marker must publish.
    peer = IdeaLake(p.lake.db_path)
    original_open = Path.open
    observations = []

    def contend_before_marker(path, mode="r", *args, **kwargs):
        if path == folder / "assessment.json" and mode == "x" and not observations:
            observations.append("actual marker create")
            before_files, before_state = _files(folder), _lifecycle(peer)
            with pytest.raises((AttemptEffectBusy, AttemptEffectInDoubt)):
                evaluator.check_active_evals({0: copy(ep)}, p.results, p.cfg, lake=peer)
            assert _files(folder) == before_files
            assert _lifecycle(peer) == before_state
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", contend_before_marker)
    try:
        events = evaluator.check_active_evals({0: ep}, p.results, p.cfg, lake=p.lake)
        assert events == [(folder.name, 0)]
        assert events[0].attempt_ref == ep.attempt_ref
        assert observations == ["actual marker create"]
        assert p.lake.get_stage_state(folder.name, "training") == "COMPLETE"
        assert p.lake.get_stage_state(folder.name, "evaluation") == "FAILED"
        assert evaluator.check_active_evals({0: ep}, p.results, p.cfg, lake=p.lake) == []
        p.popen.assert_called_once()
    finally:
        peer.close()
