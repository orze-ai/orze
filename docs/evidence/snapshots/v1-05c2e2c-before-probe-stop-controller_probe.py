"""Bounded local probes enrolled in the current controller's process tree.

``run_probe`` delegates to the current ``subprocess.run`` unchanged when no
controller is bound. Registered-controller calls support the noninteractive
argv/env/cwd/timeout/check/capture/text/DEVNULL subset used by local GPU and
kernel probes. Unsupported options fail before preparation; they never select
a raw-process fallback. A context-bound probe requires a finite positive
timeout. No input, shell, arbitrary output file or remote execution is added.

The common supervisor hooks own pending/READY/GO/tree membership. This adapter
only supplies a fresh ``controller_probe`` identity and settles the action
after TREE_CLOSED, pipe EOF, complete output accounting and local FD closure.
Full stdout plus stderr must fit 64 KiB. A stopped worker's true return code
is never rewritten, including zero; STOP, truncation and unknown closure do
not return a normal CompletedProcess. Output metadata contains hashes/counts,
not output bodies, argv or environment. This is not legacy repair admission.
"""
from __future__ import annotations

import codecs
import hashlib
import locale
import math
import os
from pathlib import Path
import secrets
import subprocess
import time

from orze.engine.bounded_executor import _Streams, _closed, _executable, _ready
from orze.engine.supervised_process import (
    SupervisionUnavailable, SupervisionUncertain, prepare_supervised,
)
from orze.engine.supervisor_worker import canonical
from orze.engine.termination_hold import TerminationUnconfirmed


MAX_OUTPUT_BYTES = 65536
STOP_TIMEOUT = 3.0
EOF_TIMEOUT = 1.0


class ControllerProbeHOLD(TerminationUnconfirmed):
    """A probe must not become a permissive None/empty inventory on HOLD."""


def _current_controller():
    # Lazy imports keep low-level GPU lease modules independent of controller
    # initialization. Import failures are not interpreted as an absent owner.
    from orze.engine.controller_control import ControllerHOLD, current_controller
    try:
        return current_controller()
    except ControllerHOLD as exc:
        raise ControllerProbeHOLD("controller_probe_context_unavailable") from exc


def _settle(process, **metadata):
    from orze.engine.controller_members import settle_process
    return settle_process(process, **metadata)


def _hold(ctx, reason):
    try:
        ctx.hold(reason)
    finally:
        raise ControllerProbeHOLD(reason) from None


def require_remote_probe_disabled():
    """Remote SSH work is not owned by the local controller supervisor."""
    ctx = _current_controller()
    if ctx is not None:
        _hold(ctx, "controller_remote_probe_unsupported")


def _inputs(popenargs, kwargs):
    options = dict(kwargs)
    if len(popenargs) == 1 and "args" not in options:
        original = popenargs[0]
    elif not popenargs and "args" in options:
        original = options.pop("args")
    else:
        raise TypeError("controller_probe_args_invalid")
    allowed = {"env", "cwd", "timeout", "check", "capture_output", "text",
               "universal_newlines", "encoding", "errors", "stdin", "stdout", "stderr"}
    if set(options) - allowed:
        raise TypeError("controller_probe_option_unsupported")
    if (type(original) not in (list, tuple) or not original
            or any(type(arg) is not str or "\0" in arg for arg in original)):
        raise ValueError("controller_probe_argv_invalid")
    budget = options.get("timeout")
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("controller_probe_timeout_invalid")
    for key in ("check", "capture_output", "text", "universal_newlines"):
        if key in options and type(options[key]) is not bool:
            raise TypeError("controller_probe_flag_invalid")
    if ("text" in options and "universal_newlines" in options
            and options["text"] != options["universal_newlines"]):
        raise subprocess.SubprocessError("controller_probe_text_flags_conflict")
    if options.get("stdin") not in (None, subprocess.DEVNULL):
        raise ValueError("controller_probe_input_unsupported")
    capture = options.get("capture_output", False)
    stdout, stderr = options.get("stdout"), options.get("stderr")
    if capture:
        if stdout is not None or stderr is not None:
            raise ValueError("stdout and stderr arguments may not be used with capture_output")
        stdout = stderr = subprocess.PIPE
    if stdout not in (subprocess.PIPE, subprocess.DEVNULL) or stderr not in (subprocess.PIPE, subprocess.DEVNULL):
        raise ValueError("controller_probe_output_target_unsupported")
    environment = options.get("env")
    if environment is None:
        environment = dict(os.environ)
    elif (type(environment) is not dict or any(type(k) is not str or not k
            or "=" in k or "\0" in k or type(v) is not str or "\0" in v
            for k, v in environment.items())):
        raise ValueError("controller_probe_environment_invalid")
    else:
        environment = dict(environment)
    cwd = str(Path(options.get("cwd") or os.getcwd()).absolute())
    encoding = options.get("encoding")
    errors = options.get("errors")
    if errors is not None and (type(errors) is not str or errors != "strict"):
        raise ValueError("controller_probe_lossy_decode_unsupported")
    text = options.get("text", options.get("universal_newlines", False)) or encoding is not None or errors is not None
    if text:
        encoding = encoding or locale.getpreferredencoding(False)
        errors = errors or "strict"
        codecs.lookup(encoding)
        codecs.lookup_error(errors)
    return {"original": original, "command": list(original), "timeout": float(budget),
            "env": environment, "cwd": cwd, "check": options.get("check", False),
            "text": text, "encoding": encoding, "errors": errors,
            "stdout": stdout, "stderr": stderr}


class _ProbeStreams(_Streams):
    """Reuse bounded pipe ownership/drain; hash every actually observed byte."""
    def __init__(self):
        self.hashers = {key: hashlib.sha256() for key in ("stdout", "stderr")}
        super().__init__()

    def drain(self, delay):
        before = dict(self.sizes)
        super().drain(delay)
        # _Streams reads at most one <=64 KiB chunk per stream per call. Its
        # retained tail therefore contains all of this call's new bytes.
        for key in before:
            count = self.sizes[key] - before[key]
            if count:
                self.hashers[key].update(self.tails[key][-count:])

    def metadata(self):
        # Hash a canonical pair of full-stream hash/count records: concatenating
        # unframed stdout and stderr would not bind the stream boundary.
        record = {key: {"sha256": self.hashers[key].hexdigest(), "bytes": count}
                  for key, count in self.sizes.items()}
        return {"output_sha256": hashlib.sha256(canonical(record)).hexdigest(),
                "output_bytes": sum(self.sizes.values())}


def _decode(raw, spec):
    if not spec["text"]:
        return raw
    return raw.decode(spec["encoding"], spec["errors"]).replace("\r\n", "\n").replace("\r", "\n")


def run_probe(*popenargs, **kwargs):
    """Run a local diagnostic; no-controller calls keep raw run semantics."""
    ctx = _current_controller()
    if ctx is None:
        return subprocess.run(*popenargs, **kwargs)
    try:
        ctx.check_admission()
    except Exception as exc:
        raise ControllerProbeHOLD("controller_probe_admission_rejected") from exc
    try:
        spec = _inputs(popenargs, kwargs)
    except Exception:
        _hold(ctx, "controller_probe_input_invalid")
    command = _executable(spec["command"], spec["env"], spec["cwd"])
    identity = {"schema": 1, "kind": "controller_probe",
                "invocation_id": secrets.token_hex(16), "scope": str(ctx.scope)}
    process = streams = binding = tree = None
    entered = stop_attempted = closed = timed_out = overflow = settled = False
    try:
        streams = _ProbeStreams()
        entered = True
        try:
            process = prepare_supervised(command, identity=identity, env=spec["env"],
                cwd=spec["cwd"], stdout=streams.writers[0], stderr=streams.writers[1])
        except SupervisionUnavailable:
            entered = False
            raise
        except SupervisionUncertain as exc:
            process = exc.process
            raise
        streams.close_writers()
        binding = _ready(process, identity, command)
        ctx.poll_control()
        # The primitive itself arbitrates a quiesce request arriving after
        # READY. False means no GO and an owned STOP, never successful start.
        start_result = process.start()
        if start_result is not None and start_result is not False:
            raise ControllerProbeHOLD("controller_probe_start_unconfirmed")
        stop_deadline = None
        if start_result is False:
            stop_attempted = True
            stop_deadline = time.monotonic() + STOP_TIMEOUT
        deadline = time.monotonic() + spec["timeout"]
        eof_deadline = None
        while True:
            phase = ctx.poll_control()
            if phase == "QUIESCING" and not stop_attempted:
                # poll() sends the owned nonblocking STOP. Continue draining;
                # sending STOP is not evidence that the tree has closed.
                stop_attempted = True
                stop_deadline = time.monotonic() + STOP_TIMEOUT
            ret = process.poll()
            if type(ret) is int:
                tree = _closed(process, binding, ret)
                closed = True
                if not streams.readers:
                    break
                eof_deadline = eof_deadline or time.monotonic() + EOF_TIMEOUT
                if time.monotonic() >= eof_deadline:
                    raise ControllerProbeHOLD("controller_probe_output_eof_unconfirmed")
            elif stop_deadline is not None and time.monotonic() >= stop_deadline:
                raise ControllerProbeHOLD("controller_probe_stop_unconfirmed")
            elif not stop_attempted and (overflow or time.monotonic() >= deadline):
                timed_out = not overflow
                stop_attempted = True
                if process.stop(timeout=STOP_TIMEOUT) is not True:
                    raise ControllerProbeHOLD("controller_probe_stop_unconfirmed")
                tree = _closed(process, binding, process.poll())
                closed = True
                continue
            streams.drain(0.01)
            overflow = overflow or sum(streams.sizes.values()) > MAX_OUTPUT_BYTES
        output = {}
        if not overflow:
            for key in ("stdout", "stderr"):
                output[key] = _decode(streams.tails[key], spec) if spec[key] == subprocess.PIPE else None
        metadata = streams.metadata()
        streams.close()
        stopped = tree["stop_requested"] or tree["forced_cleanup"]
        _settle(process, outcome="interrupted" if stopped or timed_out or overflow else "completed", **metadata)
        settled = True
        if timed_out:
            error = subprocess.TimeoutExpired(spec["original"], spec["timeout"],
                output=streams.tails["stdout"], stderr=streams.tails["stderr"])
            error.process_tree, error.returncode = tree, ret
            raise error
        if overflow:
            raise ControllerProbeHOLD("controller_probe_output_limit")
        if stopped:
            raise ControllerProbeHOLD("controller_probe_stopped")
        result = subprocess.CompletedProcess(spec["original"], ret, output["stdout"], output["stderr"])
        if spec["check"]:
            result.check_returncode()
        return result
    except BaseException as exc:
        if settled or not entered:
            raise
        # The common primitive hook already retains pre-READY uncertain
        # owners. A protocol-latched handle must not receive a second STOP.
        if process is not None and not closed and not stop_attempted and not isinstance(exc, SupervisionUncertain):
            try:
                if process.poll() is None:
                    stop_attempted = True
                    process.stop(timeout=STOP_TIMEOUT)
            except BaseException:
                pass
        _hold(ctx, "controller_probe_execution_unconfirmed")
    finally:
        if streams is not None:
            try:
                streams.close()
            except BaseException:
                _hold(ctx, "controller_probe_pipe_close_unconfirmed")
