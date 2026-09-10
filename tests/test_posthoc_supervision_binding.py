"""C2c new mechanisms: explicit protocol doubles and real SQLite publication.

No raw PID, actual process-tree closure, GPU or scientific result is inferred
from these doubles. Public real-CPU posthoc tests are a separate suite.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, create_attempt, current_attempt,
)
from orze.engine import posthoc_attempts as attempts, posthoc_supervision as proof
from orze.engine.accounting import record_compute_start
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.attempt_effect_receipts import require_closed_effects
from orze.engine.scheduler import claim
from orze.idea_lake import IdeaLake
from test_evaluation_supervision_binding import ProtocolDouble


@pytest.fixture
def case(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    idea = "idea-posthoc-proof"
    lake.insert(idea, "Posthoc", "seed: 13", "", status="queued", kind="posthoc_eval")
    results = tmp_path / "results"
    assert claim(idea, results, 0, lake=lake)
    folder = results / idea
    receipt = json.loads((folder / "claim.json").read_bytes())
    tp = SimpleNamespace(idea_id=idea, gpu=0, attempt_id=receipt["attempt_id"],
                         attempt_ref=None, process=None, is_posthoc=True,
                         start_time=time.time(), execution_identity="d" * 64)
    inputs = {"schema": 1, "kind": "posthoc_eval", "adapter": "null",
              "config_sha256": "a" * 64, "payload_sha256": "b" * 64,
              "work_dir": str(folder / "_posthoc_attempts" / tp.attempt_id / "work"),
              "execution_identity": tp.execution_identity}
    c = SimpleNamespace(lake=lake, tp=tp, folder=folder, inputs=inputs,
                        cfg={"idea_lake_db": str(lake.db_path)})
    try:
        yield c
    finally:
        lake.close()


def begin(c):
    c.tp.attempt_ref = attempts.begin(c.lake, c.tp, c.folder, launch_inputs=c.inputs)
    return c.tp.attempt_ref


def ready(c):
    ref = c.tp.attempt_ref or begin(c)
    binding = {"schema": 1, "protocol": proof.PROTOCOL,
               "identity": {"attempt_ref": asdict(ref), "scope": str(c.folder)},
               "nonce_sha256": "c" * 64, "command_sha256": "d" * 64,
               "worker": {"pid": 12001, "start_ticks": 31},
               "supervisor": {"pid": 12002, "start_ticks": 30}}
    closure = {"schema": 1, "event": "TREE_CLOSED", "binding": deepcopy(binding),
               "worker_returncode": 0, "stop_requested": False, "forced_cleanup": False,
               "reaped_children": 1, "wait_proof": "ECHILD_WALL"}
    c.tp.process = ProtocolDouble(binding, closure)
    return {"pid": 12001, "pgid": 12001, "start_ticks": 31}


def started(c):
    identity = ready(c)
    attempts.record_ready_start(c.lake, c.tp, c.folder, record_compute_start)
    attempts.started(c.lake, c.tp, c.folder, identity)


def row(c):
    return current_attempt(c.lake.conn, c.tp.idea_id, "posthoc")


def test_intent_is_peer_visible_detached_and_not_a_training_attempt(case):
    c = case
    original = deepcopy(c.inputs)
    ref = begin(c)
    c.inputs["adapter"] = "changed"
    with sqlite3.connect(c.lake.db_path) as peer:
        actual = current_attempt(peer, c.tp.idea_id, "posthoc")
        assert actual["state"] == "LAUNCHING"
        assert actual["binding"]["launch_inputs"] == original
        assert actual["binding"]["lifecycle_phase"] == "training"
        assert actual["binding"]["origin"] == "native_posthoc"
        assert actual["binding"]["process_supervision_protocol"] == proof.PROTOCOL
        assert current_attempt(peer, c.tp.idea_id, "training") is None
    assert ref.phase == "posthoc" and c.lake.get_fsm_state(c.tp.idea_id) == "CLAIMED"
    assert not (c.folder / "_compute_receipts").exists()


@pytest.mark.parametrize("phase", ["training", "evaluation"])
def test_new_posthoc_intent_cannot_overlap_an_open_other_phase(case, phase):
    c = case
    c.lake.conn.execute("BEGIN IMMEDIATE")
    create_attempt(c.lake.conn, c.tp.idea_id, phase, "other-active", {})
    c.lake.conn.commit()
    with pytest.raises(AttemptEffectBusy, match="posthoc_other_phase_active"):
        begin(c)
    assert row(c) is None
    assert c.lake.get_fsm_state(c.tp.idea_id) == "CLAIMED"


@pytest.mark.parametrize("invalid", [None, {1: "not-string-key"}, {"value": float("nan")},
                                      {"value": "x" * 65537}])
def test_launch_inputs_are_strict_bounded_objects_before_intent(case, invalid):
    c = case
    with pytest.raises(AttemptAuthorityError):
        attempts.begin(c.lake, c.tp, c.folder, launch_inputs=invalid)
    assert row(c) is None
    assert not (c.folder / "_execution_catalog.json").exists()


def test_ready_accounting_precedes_running_and_started_preserves_binding(case):
    c = case
    identity = ready(c)
    before = deepcopy(row(c)["binding"])
    attempts.record_ready_start(c.lake, c.tp, c.folder, record_compute_start)
    start = json.loads((c.folder / "_compute_receipts" / c.tp.attempt_id / "start.json").read_bytes())
    assert start["phase"] == "posthoc" and start["process_pid"] == 12001
    assert row(c)["state"] == "LAUNCHING" and c.lake.get_fsm_state(c.tp.idea_id) == "CLAIMED"
    with pytest.raises(AttemptEffectBusy, match="posthoc_attempt_not_running"):
        attempts.current(c.lake, c.tp, c.folder)
    attempts.started(c.lake, c.tp, c.folder, identity)
    actual = row(c)
    assert actual["state"] == "RUNNING"
    assert all(actual["binding"][key] == value for key, value in before.items())
    assert actual["binding"]["supervision"] == c.tp.process.binding
    assert actual["binding"]["lifecycle"]["phase_state"] == "IN_PROGRESS"
    claim_value = json.loads((c.folder / "claim.json").read_bytes())
    assert (claim_value["trainer_pid"], claim_value["trainer_start_ticks"]) == (12001, 31)
    assert attempts.current(c.lake, c.tp, c.folder) == actual


@pytest.mark.parametrize("fault", ["raw_process", "wrong_ticks", "claim_replaced"])
def test_started_cannot_publish_identity_without_matching_ready_and_claim(case, fault):
    c = case
    identity = ready(c)
    attempts.record_ready_start(c.lake, c.tp, c.folder, record_compute_start)
    if fault == "raw_process":
        c.tp.process = SimpleNamespace(pid=12001)
    elif fault == "wrong_ticks":
        identity["start_ticks"] = 32
    else:
        claim_path = c.folder / "claim.json"
        value = json.loads(claim_path.read_bytes())
        value["attempt_id"] = "replacement"
        claim_path.write_text(json.dumps(value))
    before = (c.folder / "claim.json").read_bytes()
    with pytest.raises((AttemptEffectBusy, AttemptAuthorityError)):
        attempts.started(c.lake, c.tp, c.folder, identity)
    assert row(c)["state"] == "LAUNCHING"
    assert c.lake.get_fsm_state(c.tp.idea_id) == "CLAIMED"
    assert (c.folder / "claim.json").read_bytes() == before


def test_started_sql_rejection_keeps_launching_and_latches_partial_claim(case):
    c = case
    identity = ready(c)
    attempts.record_ready_start(c.lake, c.tp, c.folder, record_compute_start)
    c.lake.conn.execute("CREATE TRIGGER reject_posthoc_running BEFORE UPDATE ON execution_attempts "
                        "WHEN NEW.state='RUNNING' BEGIN SELECT RAISE(IGNORE); END")
    with pytest.raises(AttemptEffectInDoubt):
        attempts.started(c.lake, c.tp, c.folder, identity)
    assert row(c)["state"] == "LAUNCHING"
    assert c.lake.get_fsm_state(c.tp.idea_id) == "CLAIMED"
    assert json.loads((c.folder / "claim.json").read_bytes())["trainer_pid"] == 12001
    assert (c.folder / "_attempt_effect.lock").exists()


@pytest.mark.parametrize("change", ["token_removed", "flag_changed", "missing_lake"])
def test_persisted_native_routing_cannot_fall_back_to_legacy(case, change):
    c = case
    started(c)
    if change == "token_removed":
        c.tp.attempt_ref = None
        with pytest.raises(AttemptEffectBusy, match="posthoc_native_token_required"):
            attempts.current(c.lake, c.tp, c.folder)
    else:
        if change == "flag_changed":
            c.tp.is_posthoc = False
        with pytest.raises(AttemptEffectBusy):
            attempts.require_catalog(None if change == "missing_lake" else c.lake,
                                     c.folder, c.cfg, handle=c.tp)


def test_catalog_probe_detects_posthoc_after_routing_tokens_are_removed(case):
    c = case
    started(c)
    (c.folder / "_execution_catalog.json").unlink()
    claim_path = c.folder / "claim.json"
    value = json.loads(claim_path.read_bytes())
    value.pop("lifecycle_db")
    claim_path.write_text(json.dumps(value))
    c.tp.attempt_ref = None
    with pytest.raises(AttemptEffectBusy, match="posthoc_native_catalog_required"):
        attempts.require_catalog(None, c.folder, c.cfg, handle=c.tp)


@pytest.mark.parametrize("running", [False, True])
def test_confirmed_stop_closes_attempt_not_task_and_keeps_actual_zero(case, running):
    c = case
    if running:
        started(c)
    else:
        ready(c)
    c.tp.process._test_closure["stop_requested"] = True
    before_state = c.lake.get_fsm_state(c.tp.idea_id)
    attempts.failed_launch(c.lake, c.tp, c.folder, 0)
    actual = row(c)
    assert actual["state"] == "TERMINAL" and actual["terminal"]["outcome"] == "failed"
    assert actual["terminal"]["return_code"] == 0
    assert actual["terminal"]["process_tree"]["stop_requested"] is True
    assert c.lake.get_fsm_state(c.tp.idea_id) == before_state
    receipt = json.loads((c.folder / "_compute_receipts" / c.tp.attempt_id / "terminal.json").read_bytes())
    assert receipt["phase"] == "posthoc" and receipt["return_code"] == 0
    require_closed_effects(c.folder)
    assert attempts.current(c.lake, c.tp, c.folder) is False


@pytest.mark.parametrize("possible_process", ["present", "uncertain", "none"])
def test_not_started_requires_explicit_no_process_fact(case, possible_process):
    c = case
    begin(c)
    if possible_process == "present":
        ready(c)
    elif possible_process == "uncertain":
        c.tp._termination_unconfirmed = True
    if possible_process != "none":
        with pytest.raises(AttemptEffectBusy, match="posthoc_created_process_cannot_be_not_started"):
            attempts.failed_launch(c.lake, c.tp, c.folder, None, not_started=True)
        assert row(c)["state"] == "LAUNCHING"
    else:
        attempts.failed_launch(c.lake, c.tp, c.folder, None, not_started=True)
        assert row(c)["state"] == "NOT_STARTED"
    assert not (c.folder / "_compute_receipts").exists()


def test_posthoc_proof_rejects_wrong_phase_and_stopped_zero_is_not_success(case):
    c = case
    started(c)
    c.tp.process._test_closure["stop_requested"] = True
    closure = proof.require_closed(c.tp, row(c), c.folder, 0)
    assert proof.failure_override(closure, None)[0:2] == ("failed", "posthoc_process_tree_stopped")
    c.tp.attempt_ref = AttemptRef(c.tp.idea_id, "training", c.tp.attempt_id, 1)
    with pytest.raises(AttemptEffectBusy, match="posthoc_attempt_identity_missing"):
        proof.identity(c.tp, c.folder)


def test_true_legacy_scope_without_catalog_does_not_enroll(tmp_path):
    folder = tmp_path / "results" / "idea-legacy"
    folder.mkdir(parents=True)
    attempts.require_catalog(None, folder, {}, handle=SimpleNamespace(is_posthoc=True))
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize("foreign", [False, True])
def test_artifact_binding_is_detached_and_cannot_change_results_scope(case, foreign):
    c = case
    artifact = {"contract": {"version": 1, "outputs": {}},
                "root": str(c.folder.parent.parent / "control" / "artifacts"),
                "scope": str(c.folder.parent if not foreign else c.folder.parent / "other"),
                "spec_fingerprint": "e" * 64}
    if foreign:
        with pytest.raises(AttemptEffectBusy, match="posthoc_artifact_scope_mismatch"):
            attempts.begin(c.lake, c.tp, c.folder, launch_inputs=c.inputs, artifact_binding=artifact)
        assert row(c) is None
    else:
        expected = deepcopy(artifact)
        attempts.begin(c.lake, c.tp, c.folder, launch_inputs=c.inputs, artifact_binding=artifact)
        artifact["spec_fingerprint"] = "f" * 64
        assert row(c)["binding"]["artifact_publication"] == expected
