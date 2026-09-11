"""New resolver mechanisms: real SQLite and pipes, explicit OS-proof doubles.

Pipe payloads are synthetic fixture bytes; they do not establish a real CPU
tree. Separate native tree tests exercise actual supervised resolver processes.
No absent API or configuration option is counted as a preceding-source defect.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import AttemptRef, create_attempt, current_attempt
from orze.engine import native_artifact_preflight as native, artifact_preflight_supervision as proof, process
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.execution_authority import execution_transaction
from orze.engine.failure import _reset_idea_for_retry
from orze.engine.scheduler import claim
from orze.idea_lake import IdeaLake
from supervision_fixture import SimulatedSupervisedProcess
from test_evaluation_supervision_binding import case as evaluation_case


@pytest.fixture
def resolver(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    script = tmp_path / "resolver.py"
    script.write_text("# Synthetic protocol fixture, not executed.\n")
    cfg = {"results_dir": str(results), "idea_lake_db": str(tmp_path / "lake.db"),
        "_project_root": str(tmp_path), "python": sys.executable,
        "train_extra_env": {"PRIVATE_LABEL": "fixture-secret-value"},
        "artifact_preflight": {"enabled": True, "script": str(script), "network": "inherit"}}
    lake = IdeaLake(cfg["idea_lake_db"])
    idea = "idea-preflight-proof"
    lake.insert(idea, "CPU resolver", "seed: 1\n", "", status="queued")
    assert claim(idea, results, 4, lake)
    folder = results / idea
    (folder / "idea_config.yaml").write_text("seed: 1\n")
    a = SimpleNamespace(lake=lake, cfg=cfg, results=results, folder=folder,
        idea=idea, code=0, prepared=[], args=[], output=b"\x00fixture stdout \xe9\x9b\xaa\n",
        error=b"stderr contains fixture-secret-value\n")

    def prepare(cmd, *, identity, **kwargs):
        # Actual pipe bytes, with declared synthetic READY/closure OS facts.
        os.write(kwargs["stdout"], a.output)
        os.write(kwargs["stderr"], a.error)
        child = SimpleNamespace(pid=610001 + len(a.prepared), returncode=a.code)
        child.poll = lambda: child.returncode
        child.wait = lambda timeout=None: child.returncode
        supervised = SimulatedSupervisedProcess(child, identity, cmd)
        a.prepared.append(supervised)
        a.args.append((list(cmd), deepcopy(kwargs)))
        return supervised

    monkeypatch.setattr(process, "prepare_supervised", prepare)
    a.run = lambda: native.run_native_artifact_preflight(idea, results, cfg, lake)
    try:
        yield a
    finally:
        lake.close()


@pytest.mark.parametrize("code", [0, 7])
def test_same_claim_hashes_actual_pipe_bytes_and_caches_without_gpu_or_plaintext(resolver, code):
    a = resolver
    a.code = code
    claim_before = (a.folder / "claim.json").read_bytes()
    result = a.run()
    row = current_attempt(a.lake.conn, a.idea, "artifact_preflight")
    receipt = json.loads((a.folder / "artifact_preflight.json").read_text())
    assert isinstance(result, native.ArtifactPreflightResult)
    assert bool(result) is (code == 0)
    assert result.attempt_ref.phase == "artifact_preflight"
    assert result.attempt_ref.attempt_id != json.loads(claim_before)["attempt_id"]
    assert receipt["stdout_sha256"] == hashlib.sha256(a.output).hexdigest()
    assert receipt["stderr_sha256"] == hashlib.sha256(a.error).hexdigest()
    assert receipt["attempt_ref"] == asdict(result.attempt_ref)
    assert receipt["status"] == ("passed" if code == 0 else "failed")
    assert row["terminal"]["process_tree"]["binding"]["worker"]["pid"] == a.prepared[0].pid
    assert row["binding"]["preflight_identity"] == process._artifact_preflight_identity(a.idea, a.results, a.cfg)
    assert "fixture-secret-value" not in json.dumps(row) + json.dumps(receipt)
    assert "environment_sha256" not in row["binding"]
    assert (a.folder / "claim.json").read_bytes() == claim_before
    assert a.lake.get_fsm_state(a.idea) == "CLAIMED"
    assert not (a.folder / "_compute_receipts").exists()
    assert a.args[0][1]["env"]["NVIDIA_VISIBLE_DEVICES"] == "none"
    assert all(a.args[0][1]["env"][key] == "" for key in (
        "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    a.cfg["train_extra_env"]["PRIVATE_LABEL"] = "changed-secret-is-not-a-new-execution"
    assert a.run() == result
    assert len(a.prepared) == 1
    assert current_attempt(a.lake.conn, a.idea, "artifact_preflight") == row


def test_new_claim_produces_distinct_action_without_reusing_training_claim_id(resolver):
    a = resolver
    a.code = 7
    old = a.run()
    _reset_idea_for_retry(a.folder)
    claim_id = json.loads((a.folder / "claim.json").read_text())["attempt_id"]
    a.code = 0
    new = a.run()
    assert not old and new
    assert new.attempt_ref.generation == old.attempt_ref.generation + 1
    assert new.attempt_ref.attempt_id not in (old.attempt_ref.attempt_id, claim_id)
    assert len(a.prepared) == 2


@pytest.mark.parametrize("rejection", ["missing_script", "required_offline"])
def test_static_rejection_is_confirmed_not_started_without_fake_tree_or_stream_hashes(resolver, rejection):
    a = resolver
    if rejection == "missing_script":
        a.cfg["artifact_preflight"]["script"] = str(a.folder / "missing.py")
    else:
        a.cfg["artifact_preflight"]["network"] = "required"
        a.cfg["train_extra_env"]["HF_HUB_OFFLINE"] = "1"
    result = a.run()
    row = current_attempt(a.lake.conn, a.idea, "artifact_preflight")
    receipt = json.loads((a.folder / "artifact_preflight.json").read_text())
    assert not result
    assert row["state"] == "NOT_STARTED"
    assert row["terminal"]["process_tree"] is None
    assert row["terminal"]["return_code"] is None
    assert "supervision" not in row["binding"] and "process_pid" not in row["binding"]
    assert receipt["status"] == "configuration_error"
    assert not {"stdout_sha256", "stderr_sha256", "exit_code"}.intersection(receipt)
    assert a.run() == result
    assert a.prepared == []
    assert not (a.folder / "_compute_receipts").exists()


def test_other_phase_intent_prevents_native_resolver_before_prepare(resolver):
    a = resolver
    with execution_transaction(a.lake, a.folder) as tx:
        other = create_attempt(tx.conn, a.idea, "evaluation", "other-active", {"source": "fixture"})
        tx.watch_attempt(other)
    with pytest.raises(native.ArtifactPreflightHOLD):
        a.run()
    assert a.prepared == []
    assert current_attempt(a.lake.conn, a.idea, "artifact_preflight") is None
    assert not (a.folder / "artifact_preflight.json").exists()


def test_receipt_replacement_does_not_replay_or_requalify_cached_action(resolver):
    a = resolver
    a.run()
    row = current_attempt(a.lake.conn, a.idea, "artifact_preflight")
    path = a.folder / "artifact_preflight.json"
    value = json.loads(path.read_text())
    value["stdout_sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(native.ArtifactPreflightHOLD):
        a.run()
    assert len(a.prepared) == 1
    assert current_attempt(a.lake.conn, a.idea, "artifact_preflight") == row


@pytest.mark.parametrize("fault", ["wrong_phase", "missing_closure", "stop_zero"])
def test_preflight_needs_own_closed_protocol_and_stop_zero_is_not_success(tmp_path, fault):
    handle, row, folder = evaluation_case(tmp_path)
    old = handle.attempt_ref
    handle.attempt_ref = AttemptRef(old.task_id, "artifact_preflight", old.attempt_id, old.generation)
    handle.process._test_binding["identity"]["attempt_ref"] = asdict(handle.attempt_ref)
    handle.process._test_closure["binding"] = deepcopy(handle.process._test_binding)
    row["binding"]["supervision"] = deepcopy(handle.process._test_binding)
    if fault == "wrong_phase":
        handle.attempt_ref = old
    elif fault == "missing_closure":
        handle.process._test_closure = None
    else:
        handle.process._test_closure["stop_requested"] = True
        closed = proof.require_closed(handle, row, folder, 0)
        assert closed["worker_returncode"] == 0
        assert proof.failure_override(closed, ("completed", "old", ""))[0] == "failed"
        return
    with pytest.raises(AttemptEffectBusy):
        proof.require_closed(handle, row, folder, 0)
