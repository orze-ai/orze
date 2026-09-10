"""B2 new public adapter mechanisms, not historical API-absence regressions.

Actual B1 training and evaluator launch/completion use real SQLite and files;
only subprocess/GPU observation boundaries and selected IO faults are doubled.
No provider, CPU/GPU workload, repeat task, or scientific-validity claim.
"""
import copy
import errno
import json
import os
from pathlib import Path

import pytest

from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.core.research_observations import observations_for_attempt
from orze.engine import evaluator, observation_publication
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.evaluation_retry import request_evaluation_retry, EvaluationRetryError
from orze.engine.sealed import compute_sealed_hashes, write_sealed_manifest
from test_observation_snapshot_contract import (
    project, artifact_project, native_case, _launch, _finish, _observations, _envelope,
)


def _unchanged_training(c):
    assert (c.folder / "metrics.json").read_bytes() == c.metrics_before
    assert current_attempt(c.lake.conn, c.idea, "training") == c.training_before


def test_legacy_existing_output_does_not_skip_isolated_evaluation_or_choose_its_score(project):
    c = project
    old = b'{"status":"COMPLETED","quality":999}'
    (c.folder / "assessment.json").write_bytes(old)
    script = Path(c.cfg["eval_script"])
    original_script = script.read_bytes()
    ep, _, manifest, output = _launch(c)
    cmd = c.eval_calls[-1][0]
    assert all(Path(part).is_absolute() for part in cmd[:2])
    assert Path(cmd[1]) == output.parent.parent / "entrypoint.py"
    assert Path(cmd[1]).read_bytes() == original_script
    assert Path(cmd[1]).stat().st_ino != script.stat().st_ino
    script.write_text("# changed after launch; not the executed snapshot\n")
    row = _finish(c, ep, output, _observations()[:1])
    records = observations_for_attempt(c.lake.conn, ep.attempt_ref)
    assert row["terminal"]["outcome"] == "completed"
    assert records[0]["values"]["quality"] == 0
    assert records[0]["protocol_fingerprint"] == manifest["protocol_fingerprint"]
    assert (c.folder / "assessment.json").read_bytes() == old
    assert Path(cmd[1]).read_bytes() == original_script
    _unchanged_training(c)


def test_reserved_paths_override_inherited_and_configured_environment(project, monkeypatch):
    c = project
    keys = ("ORZE_EVALUATION_INPUT_MANIFEST", "ORZE_EVALUATION_OUTPUT_DIR",
            "ORZE_EVALUATION_OUTPUT_PATH")
    c.cfg["train_extra_env"] = dict.fromkeys(keys, "/wrong/config/path")
    for key in keys:
        monkeypatch.setenv(key, "/wrong/inherited/path")
    original = evaluator.subprocess.Popen
    observed = []
    def popen(*args, **kwargs):
        observed.append(copy.deepcopy({"cwd": kwargs.get("cwd"), "env": kwargs["env"]}))
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluator.subprocess, "Popen", popen)
    ep, manifest_path, _, output = _launch(c)
    assert observed[0]["cwd"] == str(output.parent)
    assert [observed[0]["env"][key] for key in keys] == [
        str(manifest_path), str(output.parent), str(output)]
    _finish(c, ep, output, _observations()[:1])


@pytest.mark.parametrize("payload", [None, b"[]", b"{", b'{"schema":true,"observations":[]}',
    b'{"schema":1,"observations":[{"name":"bad","values":{"x":NaN},"validation":'
    b'{"status":"valid","reason_code":"author"},"comparison_scope":null}]}',
    _envelope([_observations()[0], _observations()[0]]), b" " * 65537],
    ids=["missing", "list", "invalid_json", "bool_schema", "nonfinite", "duplicate", "oversize"])
def test_bad_or_missing_envelope_closes_without_observations_or_shared_failure_marker(project, payload):
    c = project
    c.cfg["eval_output"] = "metrics.json"  # Legacy alias must be irrelevant here.
    ep, _, _, output = _launch(c)
    if payload is not None:
        output.write_bytes(payload)
    ep.process.returncode = 0
    active = {0: ep}
    result = evaluator.check_active_evals(active, c.results, c.cfg, lake=c.lake)
    assert result == [(c.idea, 0)] and not active
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "failed"
    assert row["terminal"]["artifact_ids"] == row["terminal"]["observation_ids"] == []
    assert observations_for_attempt(c.lake.conn, ep.attempt_ref) == []
    assert artifacts_for_attempt(c.lake.conn, ep.attempt_ref) == []
    assert (output.parent.parent / "failure.json").exists()
    _unchanged_training(c)


@pytest.mark.parametrize("change", ["different_protocol", "remove_contract"])
def test_running_native_contract_cannot_be_changed_or_downgraded(project, change):
    c = project
    ep, _, _, output = _launch(c)
    output.write_bytes(_envelope(_observations()[:1]))
    ep.process.returncode = 0
    if change == "remove_contract":
        c.cfg.pop("observation_contract")
    else:
        c.cfg["observation_contract"]["protocol_id"] = "different protocol"
    active = {0: ep}
    with pytest.raises(AttemptEffectBusy):
        evaluator.check_active_evals(active, c.results, c.cfg, lake=c.lake)
    assert active[0] is ep
    assert current_attempt(c.lake.conn, c.idea, "evaluation")["state"] == "RUNNING"
    assert observations_for_attempt(c.lake.conn, ep.attempt_ref) == []
    _unchanged_training(c)


def test_changed_input_snapshot_cannot_launch_but_old_worker_source_is_not_selected(project):
    c = project
    (c.folder / "best_model.pt").write_bytes(b"late original worker data")
    snapshot = Path(c.input_records[0]["path"])
    snapshot.chmod(0o600)
    snapshot.write_bytes(b"tampered accepted snapshot")
    ep = evaluator.launch_eval(c.idea, 0, c.results, c.cfg, lake=c.lake,
                               source_event=c.source_event)
    assert ep is None
    assert c.eval_calls == []
    assert current_attempt(c.lake.conn, c.idea, "evaluation") is None
    _unchanged_training(c)


def test_partial_manifest_write_never_launches_or_changes_training(project, monkeypatch):
    c = project
    write = observation_publication.os.write
    calls = []
    def fail(fd, payload):
        if os.readlink(f"/proc/self/fd/{fd}").endswith("/input_manifest.json"):
            calls.append(True)
            if len(calls) == 1:
                return write(fd, payload[:3])
            raise OSError(errno.ENOSPC, "synthetic manifest publication failure")
        return write(fd, payload)
    monkeypatch.setattr(observation_publication.os, "write", fail)
    assert evaluator.launch_eval(c.idea, 0, c.results, c.cfg, lake=c.lake,
                                  source_event=c.source_event) is None
    assert calls == [True, True] and c.eval_calls == []
    assert current_attempt(c.lake.conn, c.idea, "evaluation") is None
    _unchanged_training(c)


@pytest.mark.parametrize("when", ["before_launch", "before_acceptance"])
def test_sealed_verification_keeps_canonical_manifest_root(project, when):
    c = project
    script = Path(c.cfg["eval_script"])
    c.cfg["sealed_files"] = [str(script)]
    write_sealed_manifest(c.results, compute_sealed_hashes(c.cfg["sealed_files"]))
    if when == "before_launch":
        script.write_text("# tampered sealed source\n")
        assert evaluator.launch_eval(c.idea, 0, c.results, c.cfg, lake=c.lake,
                                      source_event=c.source_event) is None
        assert c.eval_calls == []
    else:
        ep, _, _, output = _launch(c)
        output.write_bytes(_envelope(_observations()[:1]))
        script.write_text("# tampered sealed source\n")
        ep.process.returncode = 0
        active = {0: ep}
        with pytest.raises(AttemptEffectBusy, match="sealed_files_changed"):
            evaluator.check_active_evals(active, c.results, c.cfg, lake=c.lake)
        assert active[0] is ep
        assert observations_for_attempt(c.lake.conn, ep.attempt_ref) == []
    _unchanged_training(c)


def test_explicit_retry_rejects_changed_protocol_without_reselecting_training(project):
    c = project
    ep, _, _, output = _launch(c)
    _finish(c, ep, output, _observations()[:1], code=1)
    before = current_attempt(c.lake.conn, c.idea, "evaluation")
    c.cfg["observation_contract"]["protocol_id"] = "new protocol cannot be silent retry"
    with pytest.raises(EvaluationRetryError):
        request_evaluation_retry(c.idea, c.results, c.cfg, c.lake)
    assert c.lake.get_fsm_state(c.idea) == "FAILED"
    assert current_attempt(c.lake.conn, c.idea, "evaluation") == before
    _unchanged_training(c)
