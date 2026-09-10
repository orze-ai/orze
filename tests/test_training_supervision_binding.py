"""New C2b metadata/protocol mechanisms, not old missing-API regressions.

ProtocolDouble is an explicit supervised-handle boundary double, not a real
CPU worker, SQLite publication, GPU lease or scientific-result proof. Actual
native CPU launch/completion tests belong to the separate integration suite.
"""
from copy import deepcopy
from dataclasses import asdict
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import AttemptRef
from orze.engine import evaluation_supervision, process_supervision, training_supervision as training
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.supervised_process import SupervisionUncertain
from test_evaluation_supervision_binding import ProtocolDouble, case as evaluation_case


def case(tmp_path):
    ref = AttemptRef("idea-training-proof", "training", "train-proof", 1)
    folder = tmp_path / ref.task_id
    binding = {
        "schema": 1, "protocol": training.PROTOCOL,
        "identity": {"attempt_ref": asdict(ref), "scope": str(folder)},
        "nonce_sha256": "a" * 64, "command_sha256": "b" * 64,
        "worker": {"pid": 11001, "start_ticks": 21},
        "supervisor": {"pid": 11002, "start_ticks": 20},
    }
    closure = {
        "schema": 1, "event": "TREE_CLOSED", "binding": deepcopy(binding),
        "worker_returncode": 0, "stop_requested": False, "forced_cleanup": False,
        "reaped_children": 1, "wait_proof": "ECHILD_WALL",
    }
    tp = SimpleNamespace(idea_id=ref.task_id, attempt_id=ref.attempt_id,
                         attempt_ref=ref, process=ProtocolDouble(binding, closure))
    row = {"state": "RUNNING", "binding": {
        "origin": "native_training", "process_supervision_protocol": training.PROTOCOL,
        "supervision": deepcopy(binding)}}
    return tp, row, folder


@pytest.mark.parametrize("change", ["task", "attempt", "phase", "mapping"])
def test_training_identity_requires_actual_matching_training_attempt(tmp_path, change):
    tp, _, folder = case(tmp_path)
    if change == "task":
        tp.attempt_ref = AttemptRef("other-task", "training", tp.attempt_id, 1)
    elif change == "attempt":
        tp.attempt_ref = AttemptRef(tp.idea_id, "training", "other-attempt", 1)
    elif change == "phase":
        tp.attempt_ref = AttemptRef(tp.idea_id, "evaluation", tp.attempt_id, 1)
    else:
        tp.attempt_ref = asdict(tp.attempt_ref)
    with pytest.raises(AttemptEffectBusy, match="^training_attempt_identity_missing$"):
        training.identity(tp, folder)


@pytest.mark.parametrize("change,reason", [
    ("historical", "training_supervision_unbound"),
    ("scope", "training_supervision_binding_invalid"),
    ("nonce", "training_supervision_binding_changed"),
    ("boolean_exit", "training_exit_unconfirmed"),
    ("closure_returncode", "training_process_tree_receipt_invalid"),
])
def test_training_requires_same_binding_and_strict_closed_facts(tmp_path, change, reason):
    tp, row, folder = case(tmp_path)
    ret = 0
    if change == "historical":
        row["binding"].pop("process_supervision_protocol")
    elif change == "scope":
        folder = tmp_path / "other-scope" / tp.idea_id
    elif change == "nonce":
        row["binding"]["supervision"]["nonce_sha256"] = "c" * 64
    elif change == "boolean_exit":
        ret = False
    else:
        tp.process._test_closure["worker_returncode"] = 7
    with pytest.raises(AttemptEffectBusy, match="^" + reason + "$"):
        training.require_closed(tp, row, folder, ret)


@pytest.mark.parametrize("state", ["historical", "not_stopped", "stopped"])
def test_only_protocol_bound_explicit_stop_can_bind_failed_initialization(tmp_path, state):
    tp, row, folder = case(tmp_path)
    row["state"] = "LAUNCHING"
    row["binding"].pop("supervision")
    tp.process._test_closure["stop_requested"] = state != "not_stopped"
    if state == "historical":
        row["binding"].pop("process_supervision_protocol")
    before = deepcopy(row)
    if state != "stopped":
        with pytest.raises(AttemptEffectBusy):
            training.require_closed(tp, row, folder, 0, allow_launch_cleanup=True)
    else:
        closure = training.require_closed(tp, row, folder, 0, allow_launch_cleanup=True)
        bound = training.bind_launch_cleanup(row, closure)
        assert bound["origin"] == "native_training"
        assert bound["process_pid"] == tp.process.pid == 11001
        assert bound["supervision"] == closure["binding"]
        assert bound["process_pid"] != tp.process.supervisor_pid
    assert row == before


@pytest.mark.parametrize("flag", ["stop_requested", "forced_cleanup"])
def test_training_stop_zero_is_preserved_but_cannot_be_a_success(tmp_path, flag):
    tp, row, folder = case(tmp_path)
    tp.process._test_closure[flag] = True
    closure = training.require_closed(tp, row, folder, 0)
    expected = ("failed", "training_process_tree_stopped",
                "Training required process-tree termination")
    assert training.failure_override(closure, None) == expected
    assert training.failure_override(closure, ("completed", "bad", "bad")) == expected
    interruption = ("interrupted", "interruption_timeout", "Timed out")
    assert training.failure_override(closure, interruption) == interruption
    assert closure["worker_returncode"] == tp.process.poll() == 0


def test_training_unknown_transport_latches_without_signals_or_claiming_exit(tmp_path):
    tp, row, folder = case(tmp_path)

    def uncertain():
        raise SupervisionUncertain("injected_protocol_loss", process=tp.process)

    tp.process.poll = uncertain
    before = deepcopy(row)
    with pytest.raises(AttemptEffectBusy, match="^training_supervision_unconfirmed$"):
        training.require_closed(tp, row, folder, 0)
    assert tp._termination_unconfirmed is True
    assert row == before


def test_training_normal_closed_zero_binding_is_detached(tmp_path):
    tp, row, folder = case(tmp_path)
    assert training.identity(tp, folder) == tp.process.binding["identity"]
    bound = training.ready_binding(tp, folder)
    bound["worker"]["pid"] = 99001
    assert tp.process.pid == 11001
    closure = training.require_closed(tp, row, folder, 0)
    assert training.failure_override(closure, None) is None
    assert closure["worker_returncode"] == 0


def test_shared_adapter_does_not_authorize_an_unsupported_phase(tmp_path):
    tp, _, folder = case(tmp_path)
    with pytest.raises(ValueError, match="^process_supervision_phase_invalid$"):
        process_supervision.identity(tp, folder, phase="posthoc")


def test_extracted_evaluation_public_signatures_and_errors_match_complete_old_module(tmp_path):
    snapshot = (Path(__file__).resolve().parents[1] / "docs/evidence/snapshots"
                / "v1-05c2b-before-shared-evaluation-supervision.py")
    original = {}
    exec(compile(snapshot.read_bytes(), str(snapshot), "exec"), original)
    for name in ("identity", "ready_binding", "bound_binding", "require_closed",
                 "failure_override", "bind_launch_cleanup"):
        assert inspect.signature(original[name]) == inspect.signature(getattr(evaluation_supervision, name))
    for change in ("normal", "historical", "wrong_phase", "stopped"):
        outcomes = []
        for module in (original, vars(evaluation_supervision)):
            ep, row, folder = evaluation_case(tmp_path)
            if change == "historical":
                row["binding"].pop("process_supervision_protocol")
            elif change == "wrong_phase":
                ep.attempt_ref = AttemptRef(ep.idea_id, "training", ep.attempt_id, 1)
            elif change == "stopped":
                ep.process._test_closure["stop_requested"] = True
            try:
                closure = module["require_closed"](ep, row, folder, 0)
                outcomes.append((closure, module["failure_override"](closure, None)))
            except AttemptEffectBusy as exc:
                outcomes.append((type(exc), str(exc)))
        assert outcomes[0] == outcomes[1]
