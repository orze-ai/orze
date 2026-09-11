"""Bounded synchronous legacy execution, not a native repair authority.

The existing Linux supervisor binds a private in-memory invocation, not an
AttemptRef or durable journal. This helper cannot authorize native repair,
adoption or replay after controller restart. Scope HOLDs last for this Python
process only. No database/effect lock is held while waiting for a worker.

Output is UTF-8 with universal-newline normalization. At most 64 KiB of tail
bytes per stream are retained. Truncated/undecodable output remains diagnostic
only; callers must reject output_complete=False and stopped=True, independently
of the actual worker return code. Timeout is raised only after confirmed STOP.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import secrets
import select
import stat
import subprocess
import threading
import time
import math

from orze.engine.supervised_process import (
    SupervisedProcess, SupervisionUnavailable, SupervisionUncertain,
    prepare_supervised,
)
from orze.engine.supervisor_worker import PROTOCOL, canonical
from orze.engine.termination_hold import TerminationUnconfirmed

MAX_OUTPUT_BYTES = 65536
MAX_SCOPES = 128
STOP_TIMEOUT = 3.0
EOF_TIMEOUT = 1.0
_SCOPE_LOCK = threading.Lock()
_SCOPES = {}


class BoundedExecutorHOLD(TerminationUnconfirmed):
    """Do not turn an unconfirmed execution or refused admission into False."""


@dataclass
class _Scope:
    token: str
    witness: tuple
    state: str = "active"
    process: object = None


class BoundedExecutorResult(subprocess.CompletedProcess):
    """Compatible completed process with explicit output/STOP qualification."""
    def __init__(self, args, returncode, stdout, stderr, *, output_complete,
                 stopped, process_tree, stdout_bytes, stderr_bytes):
        super().__init__(args, returncode, stdout, stderr)
        self.output_complete = output_complete
        self.stopped = stopped
        self.process_tree = process_tree
        self.stdout_bytes = stdout_bytes
        self.stderr_bytes = stderr_bytes


def _scope_name(cwd):
    return str(Path(cwd).resolve())


def _witness(scope):
    info = Path(scope).stat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("bounded_executor_cwd_not_directory")
    return info.st_dev, info.st_ino


def require_executor_scope_clear(cwd):
    """Read-only same-controller gate, including before caller early returns."""
    scope = _scope_name(cwd)
    with _SCOPE_LOCK:
        if scope in _SCOPES:
            raise BoundedExecutorHOLD("bounded_executor_scope_" + _SCOPES[scope].state)


def _reserve(scope, token, witness):
    with _SCOPE_LOCK:
        if scope in _SCOPES:
            raise BoundedExecutorHOLD("bounded_executor_scope_" + _SCOPES[scope].state)
        if len(_SCOPES) >= MAX_SCOPES:
            raise BoundedExecutorHOLD("bounded_executor_scope_limit")
        slot = _Scope(token, witness)
        _SCOPES[scope] = slot
        return slot


def _finish_scope(scope, slot, *, held):
    with _SCOPE_LOCK:
        if _SCOPES.get(scope) is not slot:
            raise BoundedExecutorHOLD("bounded_executor_scope_identity_changed")
        if held:
            slot.state = "hold"
        else:
            del _SCOPES[scope]


def _inputs(cmd, timeout, env, cwd, before_start):
    if (type(cmd) not in (list, tuple) or not cmd
            or any(type(arg) is not str or "\0" in arg for arg in cmd)
            or type(env) is not dict or any(type(key) is not str or not key or "=" in key
                or "\0" in key or type(value) is not str or "\0" in value for key, value in env.items())
            or type(timeout) not in (int, float)
            or before_start is not None and not callable(before_start)):
        raise ValueError("bounded_executor_inputs_invalid")
    budget = float(timeout)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("bounded_executor_timeout_invalid")
    scope = _scope_name(cwd)
    return list(cmd), budget, dict(env), scope


def _executable(command, environment, scope):
    """Resolve once before prepare; only an observed missing executable is ENOENT."""
    name = command[0]
    if not name:
        raise FileNotFoundError("bounded_executor_executable_missing")
    if os.path.dirname(name):
        candidates = [Path(name) if os.path.isabs(name) else Path(scope) / name]
    else:
        candidates = []
        for directory in environment.get("PATH", os.defpath).split(os.pathsep):
            root = Path(directory) if os.path.isabs(directory) else Path(scope) / directory
            candidates.append(root / name)
    present = False
    for path in candidates:
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        present = True
        if stat.S_ISREG(info.st_mode) and os.access(path, os.X_OK):
            return [str(path.absolute()), *command[1:]]
    if present:
        raise PermissionError("bounded_executor_executable_unavailable")
    raise FileNotFoundError("bounded_executor_executable_missing")


def _ready(process, identity, command):
    if not isinstance(process, SupervisedProcess):
        raise BoundedExecutorHOLD("bounded_executor_supervised_process_required")
    binding = process.binding
    if (type(binding) is not dict or set(binding) != {"schema", "protocol", "identity",
            "nonce_sha256", "command_sha256", "worker", "supervisor"}
            or type(binding["schema"]) is not int or binding["schema"] != 1
            or binding["protocol"] != PROTOCOL
            or canonical(binding["identity"]) != canonical(identity)
            or binding["command_sha256"] != hashlib.sha256(canonical(command)).hexdigest()
            or type(process.pid) is not int or binding["worker"]["pid"] != process.pid
            or binding["supervisor"]["pid"] != process.supervisor_pid):
        raise BoundedExecutorHOLD("bounded_executor_ready_changed")
    return binding


def _closed(process, binding, ret):
    actual = process.poll()
    tree = process.closure_receipt()
    if (type(ret) is not int or type(actual) is not int or actual != ret
            or type(tree) is not dict or tree.get("event") != "TREE_CLOSED"
            or tree.get("wait_proof") != "ECHILD_WALL"
            or canonical(process.binding) != canonical(binding)
            or canonical(tree.get("binding")) != canonical(binding)
            or type(tree.get("worker_returncode")) is not int or tree["worker_returncode"] != ret
            or type(tree.get("stop_requested")) is not bool or type(tree.get("forced_cleanup")) is not bool):
        raise BoundedExecutorHOLD("bounded_executor_tree_unconfirmed")
    return tree


class _Streams:
    def __init__(self):
        self.readers, self.writers = {}, []
        self.tails = {"stdout": b"", "stderr": b""}
        self.sizes = {"stdout": 0, "stderr": 0}
        try:
            for key in self.tails:
                reader, writer = os.pipe2(os.O_CLOEXEC)
                self.readers[reader] = key
                self.writers.append(writer)
                os.set_blocking(reader, False)
        except BaseException:
            self.close()
            raise

    def close_writers(self):
        while self.writers:
            os.close(self.writers.pop())

    def drain(self, delay):
        ready, _, _ = select.select(list(self.readers), [], [], delay)
        for fd in ready:
            try:
                data = os.read(fd, MAX_OUTPUT_BYTES)
            except BlockingIOError:
                continue
            if not data:
                self.readers.pop(fd)
                os.close(fd)
                continue
            key = self.readers[fd]
            self.sizes[key] += len(data)
            self.tails[key] = (self.tails[key] + data)[-MAX_OUTPUT_BYTES:]

    def output(self):
        texts, complete = {}, not self.readers
        for key, raw in self.tails.items():
            complete = complete and self.sizes[key] <= MAX_OUTPUT_BYTES
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                complete = False
                text = raw.decode("utf-8", errors="replace")
            texts[key] = text.replace("\r\n", "\n").replace("\r", "\n")
        return texts, complete

    def close(self):
        error = None
        while self.writers or self.readers:
            fd = self.writers.pop() if self.writers else self.readers.popitem()[0]
            try:
                os.close(fd)
            except OSError as exc:
                error = error or exc
        if error is not None:
            raise error


def run_bounded_executor(cmd, *, timeout, env, cwd, prepare=None, before_start=None):
    """Return only closed execution; keep old CompletedProcess/TimeoutExpired API.

    ``before_start()`` is a local admission callback after validated READY and
    before final binding/directory checks and GO. It does not grant a durable
    native action. Refusal is HOLD even after known blocked-worker cleanup.
    """
    require_executor_scope_clear(cwd)
    original, budget, environment, scope = _inputs(cmd, timeout, env, cwd, before_start)
    witness = _witness(scope)
    command = _executable(original, environment, scope)
    token = secrets.token_hex(16)
    slot = _reserve(scope, token, witness)
    identity = {"schema": 1, "kind": "bounded_executor", "invocation_id": token, "scope": scope}
    process, streams, binding = None, None, None
    prepare_entered = started = stop_attempted = closed = False
    admission_refused = held = timed_out = False
    try:
        streams = _Streams()
        prepare_entered = True
        try:
            process = (prepare_supervised if prepare is None else prepare)(command,
                identity=identity, env=dict(environment), cwd=scope,
                stdout=streams.writers[0], stderr=streams.writers[1])
        except SupervisionUnavailable:
            prepare_entered = False  # Explicit primitive no-OS-side-effect contract.
            raise
        except SupervisionUncertain as exc:
            slot.process = exc.process
            raise
        slot.process = process
        streams.close_writers()
        binding = _ready(process, identity, command)
        if before_start is not None:
            try:
                before_start()
            except BaseException:
                admission_refused = True
                raise
        if _witness(scope) != witness or canonical(_ready(process, identity, command)) != canonical(binding):
            raise BoundedExecutorHOLD("bounded_executor_launch_identity_changed")
        process.start()
        started = True
        deadline, eof_deadline = time.monotonic() + budget, None
        while True:
            ret = process.poll()
            if type(ret) is int:
                tree = _closed(process, binding, ret)
                closed = True
                if not streams.readers:
                    break
                if eof_deadline is None:
                    eof_deadline = time.monotonic() + EOF_TIMEOUT
                if time.monotonic() >= eof_deadline:
                    raise BoundedExecutorHOLD("bounded_executor_output_eof_unconfirmed")
            elif time.monotonic() >= deadline:
                if stop_attempted:
                    raise BoundedExecutorHOLD("bounded_executor_stop_unconfirmed")
                stop_attempted = True
                if process.stop(timeout=STOP_TIMEOUT) is not True:
                    raise BoundedExecutorHOLD("bounded_executor_stop_unconfirmed")
                ret = process.poll()
                tree = _closed(process, binding, ret)
                closed = True
                timed_out = True
                continue
            streams.drain(0.01)
        if _witness(scope) != witness:
            raise BoundedExecutorHOLD("bounded_executor_scope_changed")
        texts, complete = streams.output()
        result = BoundedExecutorResult(original, ret, texts["stdout"], texts["stderr"],
            output_complete=complete, stopped=tree["stop_requested"] or tree["forced_cleanup"],
            process_tree=tree, stdout_bytes=streams.sizes["stdout"], stderr_bytes=streams.sizes["stderr"])
        if timed_out:
            error = subprocess.TimeoutExpired(["<bounded-executor>"], budget,
                output=result.stdout, stderr=result.stderr)
            error.process_tree, error.returncode = tree, ret
            error.output_complete, error.stopped = complete, result.stopped
            raise error
        return result
    except subprocess.TimeoutExpired:
        if not (closed and timed_out):
            held = True
            raise BoundedExecutorHOLD("bounded_executor_timeout_unconfirmed") from None
        raise
    except BaseException as exc:
        if prepare_entered:
            held = True
            # A latched supervisor uncertainty is not another STOP opportunity.
            if not isinstance(exc, SupervisionUncertain) and process is not None and not closed and not stop_attempted:
                try:
                    if not isinstance(process, SupervisedProcess):
                        raise BoundedExecutorHOLD("bounded_executor_process_unconfirmed")
                    ret = process.poll()
                    if ret is None:
                        stop_attempted = True
                        if process.stop(timeout=STOP_TIMEOUT) is not True:
                            raise BoundedExecutorHOLD("bounded_executor_stop_unconfirmed")
                        ret = process.poll()
                    if binding is not None:
                        _closed(process, binding, ret)
                        closed = True
                except BaseException:
                    pass
            # A rejected callback did not grant GO. Proven blocked cleanup
            # releases this active slot, but still never returns ordinary False.
            if admission_refused and not started and closed:
                held = False
            raise BoundedExecutorHOLD("bounded_executor_admission_rejected" if admission_refused
                                      else "bounded_executor_execution_unconfirmed") from None
        raise
    finally:
        close_error = None
        if streams is not None:
            try:
                streams.close()
            except BaseException as exc:
                held, close_error = prepare_entered, exc
        _finish_scope(scope, slot, held=held)
        if close_error is not None:
            if prepare_entered:
                raise BoundedExecutorHOLD("bounded_executor_pipe_close_unconfirmed") from None
            raise close_error
