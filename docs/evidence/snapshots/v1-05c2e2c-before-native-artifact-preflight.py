"""Claim-bound CPU resolver, with real stream hashes and closed-tree authority.

No GPU allocation, script sandbox, restart adoption or scientific publication.
Secret environment values are detached only in memory, never fingerprinted in
durable metadata. Cached results describe the earlier closed invocation, not
execution of a changed secret. Static rejection never crosses prepare/READY.
Large input hashing, pipe draining and process waits occur outside the writer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import stat
import time
from types import SimpleNamespace

from orze.core.execution_attempts import (
    AttemptRef, _json, create_attempt, current_attempt, finish_attempt,
    mark_running, require_current,
)
from orze.core.fs import atomic_write
from orze.engine import artifact_preflight_supervision as proof
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.execution_authority import canonical_identity_equal as same, execution_transaction
from orze.engine.supervised_process import SupervisedProcess, SupervisionUncertain
from orze.engine.termination_hold import require_no_unconfirmed_stop, terminate_execution
from orze.engine.training_attempts import _claim, _launch_state, _read, require_catalog

PHASE = "artifact_preflight"


class ArtifactPreflightHOLD(AttemptEffectInDoubt):
    """No confirmed boolean, retry, training or failure-report permission."""


@dataclass(frozen=True)
class ArtifactPreflightResult:
    success: bool
    attempt_ref: AttemptRef

    def __bool__(self):
        return self.success


def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _scope(lake, folder, cfg):
    """Structural scope only: usable by historical receipt/report consumers."""
    require_catalog(lake, folder, cfg)
    if lake is not None:
        return
    from orze.reporting.evidence import report_lifecycle_db_path
    from orze.core.evaluation_retry_state import open_existing_lake
    path = report_lifecycle_db_path(folder.parent, cfg)
    if path.exists() or path.is_symlink():
        opened = None
        try:
            opened = open_existing_lake(path)
            if current_attempt(opened.conn, folder.name, PHASE) is not None:
                raise ArtifactPreflightHOLD("artifact_preflight_catalog_required")
        finally:
            if opened is not None:
                opened.close()


def _admission(lake, folder, cfg):
    from orze.engine import launcher
    _scope(lake, folder, cfg)
    if lake is None:
        raise ArtifactPreflightHOLD("artifact_preflight_catalog_required")
    launcher._assert_launch_authorized(folder.name, folder.parent, cfg)
    launcher._assert_controller_runtime_attested(cfg)
    require_no_unconfirmed_stop(folder)


def _claim_binding(lake, folder):
    claim, _ = _read(folder / "claim.json", 8192)
    claim_id, gpu = claim.get("attempt_id"), claim.get("gpu")
    AttemptRef(folder.name, PHASE, claim_id, 1)
    if type(gpu) is not int or gpu < 0:
        raise ArtifactPreflightHOLD("artifact_preflight_claim_invalid")
    _, digest = _claim(SimpleNamespace(attempt_id=claim_id, gpu=gpu), folder, lake)
    return {"claim_attempt_id": claim_id, "claim_sha256": digest,
            "launch_lifecycle": _launch_state(lake, folder.name)}


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _hash_file(path):
    """Hash a stable regular input, never a FIFO; no new input-size policy."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return None, None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            return None, _file_identity(before)
        digest, size = hashlib.sha256(), 0
        while size <= before.st_size:
            data = os.read(fd, min(1048576, before.st_size + 1 - size))
            if not data:
                break
            digest.update(data)
            size += len(data)
        identity = _file_identity(before)
        if (size != before.st_size or _file_identity(os.fstat(fd)) != identity
                or _file_identity(path.stat()) != identity):
            raise ArtifactPreflightHOLD("artifact_preflight_input_changed")
        return digest.hexdigest(), identity
    finally:
        os.close(fd)


def _files_current(files):
    for path, expected in files:
        try:
            actual = _file_identity(path.stat())
        except FileNotFoundError:
            actual = None
        if actual != expected:
            raise ArtifactPreflightHOLD("artifact_preflight_input_changed")


def _capture(folder, cfg):
    """Legacy identity fields plus private detached invocation and stat fences."""
    from orze.engine import process as owner
    from orze.engine.launcher import _format_args
    spec = cfg.get(PHASE)
    if type(spec) is not dict or spec.get("enabled") is not True:
        raise ArtifactPreflightHOLD("artifact_preflight_configuration_invalid")
    timeout = spec.get("timeout", 300)
    if type(timeout) not in (int, float):
        raise ArtifactPreflightHOLD("artifact_preflight_timeout_invalid")
    try:
        timeout = float(timeout)
    except (ValueError, OverflowError) as exc:
        raise ArtifactPreflightHOLD("artifact_preflight_timeout_invalid") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ArtifactPreflightHOLD("artifact_preflight_timeout_invalid")
    args, script = spec.get("args", []), spec.get("script", "")
    interpreter = spec.get("interpreter", cfg.get("python", owner.sys.executable)) or ""
    if (type(args) is not list or any(type(x) is not str or "\0" in x for x in args)
            or type(script) is not str or "\0" in script
            or type(interpreter) is not str or "\0" in interpreter):
        raise ArtifactPreflightHOLD("artifact_preflight_inputs_invalid")
    root = Path(cfg.get("_project_root", ".")).absolute()
    path = Path(script) if Path(script).is_absolute() else root / script
    config = folder / "idea_config.yaml"
    if not config.exists():
        base = Path(cfg.get("base_config", "configs/base.yaml"))
        config = base if base.is_absolute() else root / base
    policy = str(spec.get("network", "inherit")).strip().lower()
    raw_extra = cfg.get("train_extra_env") or {}
    extra = raw_extra if type(raw_extra) is dict else {}
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in extra.items()})
    flags = {k: owner._truthy_env(str(extra.get(k, os.environ.get(k, ""))))
             for k in owner._OFFLINE_ENV_KEYS}
    contract = {"script": script, "args": list(args), "interpreter": interpreter,
        "network": policy, "train_extra_env_keys": sorted(str(k) for k in extra),
        "train_extra_env_type": type(raw_extra).__name__, "offline_flags": flags}
    script_sha, script_stat = _hash_file(path)
    config_sha, config_stat = _hash_file(config)
    identity = {"network_policy": policy, "script_sha256": script_sha,
        "config_sha256": config_sha, "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    rejection = None
    if policy not in ("inherit", "required", "offline"):
        rejection = {"reason": "network_policy_invalid"}
    elif not script or script_sha is None:
        rejection = {"reason": "script_missing"}
    elif config_sha is None:
        rejection = {"reason": "config_missing"}
    elif type(raw_extra) is not dict:
        rejection = {"reason": "train_extra_env_not_mapping"}
    elif policy == "required" and any(flags.values()):
        rejection = {"reason": "offline_flags_conflict_with_required_network",
                     "conflicting_env_keys": [k for k, value in flags.items() if value]}
    if policy == "offline":
        env.update({key: "1" for key in owner._OFFLINE_ENV_KEYS})
    env.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none",
        HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", ORZE_ARTIFACT_PREFLIGHT="1",
        ORZE_ARTIFACT_NETWORK_POLICY=policy, ORZE_IDEA_ID=folder.name,
        ORZE_RESULTS_DIR=str(folder.parent), ORZE_IDEA_CONFIG=str(config))
    variables = {"idea_id": folder.name, "results_dir": str(folder.parent),
                 "config": str(config), "project_root": str(root)}
    cmd = ([interpreter] if interpreter else []) + [str(path)] + _format_args(args, variables)
    if any(not key or "=" in key or "\0" in key or "\0" in value for key, value in env.items()):
        raise ArtifactPreflightHOLD("artifact_preflight_environment_invalid")
    private = json.loads(_json({"command": cmd, "environment": env, "cwd": str(root)}))
    return SimpleNamespace(**private, timeout=timeout, identity=identity, rejection=rejection,
        files=((path, script_stat), (config, config_stat)))


def _owned(lake, folder, ref, binding, *, states):
    row = require_current(lake.conn, ref, states=states)
    if (not same({key: row["binding"].get(key) for key in binding}, binding)
            or not same(_claim_binding(lake, folder), {key: binding[key] for key in
                ("claim_attempt_id", "claim_sha256", "launch_lifecycle")})):
        raise ArtifactPreflightHOLD("artifact_preflight_authority_changed")
    if row["state"] in ("RUNNING", "TERMINAL"):
        ready = row["binding"].get("supervision") or {}
        worker = ready.get("worker") if type(ready) is dict else None
        pid = row["binding"].get("process_pid")
        if (type(pid) is not int or type(worker) is not dict
                or type(worker.get("pid")) is not int or pid != worker["pid"]):
            raise ArtifactPreflightHOLD("artifact_preflight_process_identity_changed")
    return row


def _cached(row, folder, *, pending=None):
    """Historical proof, independent of later legitimate claim-byte additions."""
    from orze.engine.attempt_effect_receipts import _scan, _decode, _read as read_effect
    ref = AttemptRef(row["task_id"], PHASE, row["attempt_id"], row["generation"])
    terminal, bound = row["terminal"], row["binding"]
    if (row["state"] not in ("TERMINAL", "NOT_STARTED") or type(terminal) is not dict
            or bound.get("origin") != "native_artifact_preflight"
            or bound.get("scope") != str(folder)
            or bound.get("process_supervision_protocol") != proof.PROTOCOL):
        raise ArtifactPreflightHOLD("artifact_preflight_result_unconfirmed")
    tree, ret = terminal.get("process_tree"), terminal.get("return_code")
    static = row["state"] == "NOT_STARTED"
    if static:
        if (tree is not None or ret is not None or "supervision" in bound
                or "process_pid" in bound or terminal.get("outcome") != "configuration_error"):
            raise ArtifactPreflightHOLD("artifact_preflight_not_started_invalid")
    else:
        # The same strict historical tree schema, without claim/FSM checks.
        ready = bound.get("supervision")
        worker = ready.get("worker") if type(ready) is dict else None
        if (type(bound.get("process_pid")) is not int or type(worker) is not dict
                or type(worker.get("pid")) is not int or bound["process_pid"] != worker["pid"]
                or type(tree) is not dict or set(tree) != {"schema", "event", "binding",
                    "worker_returncode", "stop_requested", "forced_cleanup", "reaped_children", "wait_proof"}
                or type(tree.get("schema")) is not int or tree["schema"] != 1
                or tree["event"] != "TREE_CLOSED" or tree["wait_proof"] != "ECHILD_WALL"
                or type(tree["reaped_children"]) is not int or tree["reaped_children"] < 1
                or type(tree["stop_requested"]) is not bool or type(tree["forced_cleanup"]) is not bool
                or type(ret) is not int or type(tree["worker_returncode"]) is not int
                or tree["worker_returncode"] != ret or not same(tree["binding"], ready)
                or ready.get("protocol") != proof.PROTOCOL
                or ready.get("command_sha256") != bound.get("command_sha256")
                or not same(ready.get("identity"), {"attempt_ref": asdict(ref), "scope": str(folder)})):
            raise ArtifactPreflightHOLD("artifact_preflight_cached_tree_invalid")
    digest = terminal.get("effect_receipt_sha256")
    if _scan(folder, pending=pending).get(ref.attempt_id) != (digest, True):
        raise ArtifactPreflightHOLD("artifact_preflight_cached_effect_unconfirmed")
    prepared = _decode(read_effect(folder / "_execution_effects" / ref.attempt_id / "prepared.json"))
    expected = {"operation": "artifact_preflight_terminal", **{key: terminal[key] for key in
        ("outcome", "reason_code", "return_code", "process_tree", "receipt_sha256")}}
    if not same(prepared["plan"], expected):
        raise ArtifactPreflightHOLD("artifact_preflight_cached_result_changed")
    receipt, receipt_sha = _read(folder / "artifact_preflight.json")
    success = terminal["outcome"] == "completed"
    status = "configuration_error" if static else ("passed" if success else (
        "timed_out" if terminal["reason_code"] == "artifact_preflight_timeout" else "failed"))
    if (receipt_sha != terminal["receipt_sha256"] or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1 or receipt.get("idea_id") != folder.name
            or not same(receipt.get("attempt_ref"), asdict(ref))
            or receipt.get("gpu_visibility") != "hidden" or receipt.get("status") != status
            or not same({key: receipt.get(key) for key in bound["preflight_identity"]}, bound["preflight_identity"])
            or static and any(key in receipt for key in ("exit_code", "stdout_sha256", "stderr_sha256"))
            or not static and (type(receipt.get("exit_code")) is not int or receipt["exit_code"] != ret
                or any(type(receipt.get(key)) is not str or re.fullmatch(r"[0-9a-f]{64}", receipt[key]) is None
                       for key in ("stdout_sha256", "stderr_sha256")))
            or terminal["outcome"] not in ("completed", "failed", "interrupted", "configuration_error")
            or success and (ret != 0 or tree["stop_requested"] or tree["forced_cleanup"])):
        raise ArtifactPreflightHOLD("artifact_preflight_receipt_invalid")
    return success


class _Streams:
    """Two caller-owned nonblocking readers; never retain plaintext output."""
    def __init__(self):
        self.readers, self.writers = {}, []
        self.hashes = {key: hashlib.sha256() for key in ("stdout", "stderr")}
        try:
            for key in self.hashes:
                reader, writer = os.pipe2(os.O_CLOEXEC)
                self.readers[reader] = key
                self.writers.append(writer)
                os.set_blocking(reader, False)
        except BaseException:
            self.close()
            raise

    def close_writers(self):
        while self.writers:
            os.close(self.writers.pop())  # Transfer ownership before close; never retry a reused FD.

    def drain(self, delay):
        ready, _, _ = select.select(list(self.readers), [], [], delay)
        for fd in ready:
            try:
                data = os.read(fd, 65536)
            except BlockingIOError:
                continue
            if data:
                self.hashes[self.readers[fd]].update(data)
            else:
                self.readers.pop(fd)
                os.close(fd)

    def close(self):
        try:
            self.close_writers()
        finally:
            while self.readers:
                fd, _ = self.readers.popitem()
                os.close(fd)


def _monitor(handle, folder, streams, timeout, owner):
    deadline, eof_deadline, timed_out = time.monotonic() + timeout, None, False
    while True:
        ret = handle.process.poll()
        if type(ret) is int:
            if not streams.readers:
                return ret, timed_out
            if eof_deadline is None:
                eof_deadline = time.monotonic() + 1
            if time.monotonic() >= eof_deadline:
                raise ArtifactPreflightHOLD("artifact_preflight_output_eof_unconfirmed")
        elif time.monotonic() >= deadline:
            if handle._stop_attempted:
                raise ArtifactPreflightHOLD("artifact_preflight_stop_unconfirmed")
            handle._stop_attempted = True
            terminate_execution(handle, folder, phase=PHASE, reaper=owner._terminate_and_reap, timeout=1)
            timed_out = True
            continue
        streams.drain(0.01)


def _publish(lake, folder, ref, binding, captured, handle, ret, closure, receipt, outcome, reason):
    from orze.engine.attempt_effect_receipts import _sync
    static = handle is None
    _json(receipt)
    raw = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    terminal = {"outcome": outcome, "reason_code": reason, "return_code": ret,
        "process_tree": closure, "receipt_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    with execution_transaction(lake, folder) as tx:
        _scope(lake, folder, {})
        _files_current(captured.files)
        row = _owned(lake, folder, ref, binding, states=("LAUNCHING",) if static else ("RUNNING",))
        if not static and not same(closure, proof.require_closed(handle, row, folder, ret)):
            raise ArtifactPreflightHOLD("artifact_preflight_process_tree_changed")
        digest = tx.prepare(ref, {"operation": "artifact_preflight_terminal", **terminal})
        atomic_write(folder / "artifact_preflight.json", raw)
        actual, actual_sha = _read(folder / "artifact_preflight.json")
        if actual_sha != terminal["receipt_sha256"] or not same(actual, receipt):
            raise ArtifactPreflightHOLD("artifact_preflight_receipt_write_unconfirmed")
        _sync(folder)
        terminal.update(effect_receipt_sha256=digest, launch_lifecycle=binding["launch_lifecycle"])
        if finish_attempt(tx.conn, ref, terminal, not_started=static) != "committed":
            raise ArtifactPreflightHOLD("artifact_preflight_terminal_not_new")
        _owned(lake, folder, ref, binding, states=("NOT_STARTED",) if static else ("TERMINAL",))
        _files_current(captured.files)
        tx.watch_attempt(ref)
    row = _owned(lake, folder, ref, binding, states=("NOT_STARTED",) if static else ("TERMINAL",))
    return ArtifactPreflightResult(_cached(row, folder), ref)


def run_native_artifact_preflight(idea_id, results_dir, cfg, lake):
    """Execute one native CPU action or return its exact confirmed cache."""
    from orze.engine import process as owner
    from orze.engine.claim_authority import _closed
    from orze.engine.execution_catalog import bind_catalog
    folder = Path(results_dir).absolute() / idea_id
    handle, streams, tree_closed = None, None, False
    started = time.time()
    try:
        _admission(lake, folder, cfg)
        captured = _capture(folder, cfg)
        binding = {"origin": "native_artifact_preflight", "scope": str(folder),
            **_claim_binding(lake, folder), "preflight_identity": captured.identity,
            "command_sha256": _sha(captured.command), "timeout_seconds": captured.timeout,
            "cwd": captured.cwd, "process_supervision_protocol": proof.PROTOCOL}
        action_id = _sha({"phase": PHASE, "task_id": idea_id,
                         "claim_attempt_id": binding["claim_attempt_id"]})[:32]
        with execution_transaction(lake, folder) as tx:
            _scope(lake, folder, cfg)
            _closed(tx.conn, idea_id)
            _files_current(captured.files)
            previous = current_attempt(tx.conn, idea_id, PHASE)
            if previous is not None and previous["attempt_id"] == action_id:
                ref = AttemptRef(idea_id, PHASE, action_id, previous["generation"])
                row = _owned(lake, folder, ref, binding, states=("TERMINAL", "NOT_STARTED"))
                return ArtifactPreflightResult(_cached(row, folder), ref)
            bind_catalog(lake, folder, tx.lease)
            ref = create_attempt(tx.conn, idea_id, PHASE, action_id, binding)
            _owned(lake, folder, ref, binding, states=("LAUNCHING",))
            tx.watch_attempt(ref)
        receipt = {"schema_version": 1, "idea_id": idea_id, "attempt_ref": asdict(ref),
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "gpu_visibility": "hidden", **captured.identity}
        if captured.rejection is not None:
            receipt.update(status="configuration_error", **captured.rejection)
            ret, closure, outcome, reason = None, None, "configuration_error", "artifact_preflight_configuration_error"
        else:
            handle = SimpleNamespace(idea_id=idea_id, attempt_id=action_id, attempt_ref=ref,
                                     process=None, _stop_attempted=False)
            streams = _Streams()
            _admission(lake, folder, cfg)
            _files_current(captured.files)
            _owned(lake, folder, ref, binding, states=("LAUNCHING",))
            try:
                handle.process = owner.prepare_supervised(list(captured.command),
                    identity=proof.identity(handle, folder), env=dict(captured.environment),
                    cwd=captured.cwd, stdout=streams.writers[0], stderr=streams.writers[1])
            except SupervisionUncertain:
                handle._termination_unconfirmed = True
                raise
            streams.close_writers()
            ready = proof.ready_binding(handle, folder)
            if ready["command_sha256"] != binding["command_sha256"]:
                raise ArtifactPreflightHOLD("artifact_preflight_command_changed")
            with execution_transaction(lake, folder) as tx:
                _scope(lake, folder, cfg)
                _files_current(captured.files)
                _owned(lake, folder, ref, binding, states=("LAUNCHING",))
                mark_running(tx.conn, ref, {**binding, "process_pid": handle.process.pid, "supervision": ready})
                tx.watch_attempt(ref)
            _admission(lake, folder, cfg)
            fresh = _capture(folder, cfg)  # Large hashes outside writer; private env never persisted.
            if not same({key: getattr(fresh, key) for key in ("command", "environment", "cwd", "timeout", "identity")},
                        {key: getattr(captured, key) for key in ("command", "environment", "cwd", "timeout", "identity")}):
                raise ArtifactPreflightHOLD("artifact_preflight_launch_inputs_changed")
            row = _owned(lake, folder, ref, binding, states=("RUNNING",))
            proof.bound_binding(handle, row, folder)
            handle.process.start()
            ret, timed_out = _monitor(handle, folder, streams, captured.timeout, owner)
            row = _owned(lake, folder, ref, binding, states=("RUNNING",))
            closure = proof.require_closed(handle, row, folder, ret)
            tree_closed = True
            outcome, reason = (("interrupted", "artifact_preflight_timeout") if timed_out else (
                ("completed", "artifact_preflight_completed") if ret == 0 else ("failed", "artifact_preflight_nonzero")))
            outcome, reason = proof.failure_override(closure, (outcome, reason, ""))[:2]
            receipt.update(status="timed_out" if timed_out else ("passed" if outcome == "completed" else "failed"),
                exit_code=ret, **{key + "_sha256": digest.hexdigest() for key, digest in streams.hashes.items()})
        receipt.update(finished_at=datetime.now(timezone.utc).isoformat(),
                       duration_seconds=round(time.time() - started, 3))
        return _publish(lake, folder, ref, binding, captured, handle, ret, closure, receipt, outcome, reason)
    except BaseException as exc:
        if (handle is not None and handle.process is not None and not tree_closed
                and not handle._stop_attempted and getattr(handle, "_termination_unconfirmed", False) is not True):
            try:
                if not isinstance(handle.process, SupervisedProcess):
                    raise ArtifactPreflightHOLD("artifact_preflight_supervised_process_required")
                if handle.process.poll() is None:
                    handle._stop_attempted = True
                    terminate_execution(handle, folder, phase=PHASE, reaper=owner._terminate_and_reap, timeout=1)
            except BaseException as stop_exc:
                raise ArtifactPreflightHOLD("artifact_preflight_stop_unconfirmed") from stop_exc
        raise ArtifactPreflightHOLD("artifact_preflight_action_unconfirmed") from exc
    finally:
        if streams is not None:
            try:
                streams.close()
            except OSError as exc:
                raise ArtifactPreflightHOLD("artifact_preflight_pipe_close_unconfirmed") from exc
