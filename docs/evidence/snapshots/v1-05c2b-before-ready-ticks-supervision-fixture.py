"""Explicit C2a migration boundary for pre-supervisor OS-double fixtures.

Install only in tests that intentionally used fake Popen. This is not an
autouse fixture or production fallback. It preserves their OS-double behavior
while declaring the new supervision identity and closure assumptions. Real CPU
tests explicitly restore the real prepare_supervised entry point.
"""
from copy import copy, deepcopy
import hashlib
import itertools

from orze.engine import evaluator
from orze.engine.supervised_process import SupervisedProcess
from orze.engine.supervisor_worker import PROTOCOL, canonical

_SEQUENCE = itertools.count(100000)


class SimulatedSupervisedProcess(SupervisedProcess):
    def __init__(self, wrapped, identity, cmd):
        object.__setattr__(self, "_sim_wrapped", wrapped)
        raw_pid = getattr(wrapped, "pid", None)
        pid = raw_pid if type(raw_pid) is int and raw_pid > 0 else next(_SEQUENCE)
        object.__setattr__(self, "_sim_pid", pid)
        object.__setattr__(self, "_sim_started", False)
        object.__setattr__(self, "_sim_stopped", False)
        object.__setattr__(self, "_sim_binding", {
            "schema": 1, "protocol": PROTOCOL, "identity": deepcopy(identity),
            "nonce_sha256": hashlib.sha256(str(pid).encode()).hexdigest(),
            "command_sha256": hashlib.sha256(canonical(list(cmd))).hexdigest(),
            "worker": {"pid": pid, "start_ticks": 2},
            "supervisor": {"pid": pid + 1000000, "start_ticks": 1},
        })

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_sim_wrapped"), name)

    def __copy__(self):
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        object.__setattr__(result, "_sim_wrapped", copy(self._sim_wrapped))
        object.__setattr__(result, "_sim_binding", deepcopy(self._sim_binding))
        return result

    def __setattr__(self, name, value):
        if name.startswith("_sim_"):
            object.__setattr__(self, name, value)
        elif name == "pid":
            # Preserve old tests that copy a handle and substitute only its
            # observed PID; the original handle and stored binding stay fixed.
            object.__setattr__(self, "_sim_pid", value)
            setattr(self._sim_wrapped, name, value)
        else:
            setattr(self._sim_wrapped, name, value)

    @property
    def pid(self):
        return self._sim_pid

    @property
    def supervisor_pid(self):
        return self._sim_binding["supervisor"]["pid"]

    @property
    def returncode(self):
        return getattr(self._sim_wrapped, "returncode", None)

    @property
    def binding(self):
        return deepcopy(self._sim_binding)

    def start(self):
        assert not self._sim_started
        self._sim_started = True

    def poll(self):
        return self._sim_wrapped.poll()

    def wait(self, timeout=None):
        return self._sim_wrapped.wait(timeout=timeout)

    def stop(self, timeout=10):
        self._sim_stopped = True
        if self.poll() is None:
            self._sim_wrapped.returncode = -15
        return True

    def closure_receipt(self):
        code = self.poll()
        if type(code) is not int:
            return None
        return {"schema": 1, "event": "TREE_CLOSED", "binding": self.binding,
                "worker_returncode": code, "stop_requested": self._sim_stopped,
                "forced_cleanup": False, "reaped_children": 1, "wait_proof": "ECHILD_WALL"}


def install(monkeypatch):
    def prepare(cmd, *, identity, **kwargs):
        child = evaluator.subprocess.Popen(cmd, preexec_fn=evaluator._new_process_group, **kwargs)
        return SimulatedSupervisedProcess(child, identity, cmd)

    previous_reaper = evaluator._terminate_and_reap
    def reap(process, *args, **kwargs):
        from orze.engine.process import _terminate_and_reap
        target = (process._sim_wrapped if isinstance(process, SimulatedSupervisedProcess)
                  and previous_reaper is not _terminate_and_reap else process)
        confirmed = previous_reaper(target, *args, **kwargs)
        if confirmed is True and isinstance(process, SimulatedSupervisedProcess):
            process._sim_stopped = True
        return confirmed

    monkeypatch.setattr(evaluator, "prepare_supervised", prepare)
    monkeypatch.setattr(evaluator, "_terminate_and_reap", reap)


def adapt_training_reaper(monkeypatch, module, *, unwrap=True):
    """Explicitly adapt a selected OS-double reaper, including later overrides.

    Existing fake reapers may assert the original Popen identity. Their strict
    True/False result is unchanged; only True declares the simulated stop.
    Never install this adapter on a real-CPU fixture's restored reaper.
    """
    previous_reaper = module._terminate_and_reap

    def reap(process, *args, **kwargs):
        simulated = isinstance(process, SimulatedSupervisedProcess)
        target = process._sim_wrapped if simulated and unwrap else process
        confirmed = previous_reaper(target, *args, **kwargs)
        if confirmed is True and simulated:
            process._sim_stopped = True
        return confirmed

    monkeypatch.setattr(module, "_terminate_and_reap", reap)


def install_training(monkeypatch):
    """Opt a pre-existing fake-training-Popen fixture into simulated READY.

    The captured worker ticks come from that fixture's identity boundary, not
    the evaluation double's historical fixed ticks. No actual tree proof is
    asserted by these fixtures; actual CPU tests restore the real entry point.
    """
    from orze.engine import launcher, process

    def prepare(cmd, *, identity, worker_only_fds=(), **kwargs):
        kwargs["pass_fds"] = tuple(kwargs.get("pass_fds", ())) + tuple(worker_only_fds)
        child = launcher.subprocess.Popen(cmd, preexec_fn=launcher._new_process_group, **kwargs)
        handle = SimulatedSupervisedProcess(child, identity, cmd)
        captured = launcher.capture_process_identity(handle.pid)
        handle._sim_binding["worker"]["start_ticks"] = captured["start_ticks"]
        return handle

    unwrap = launcher._terminate_and_reap is not process._terminate_and_reap
    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    adapt_training_reaper(monkeypatch, launcher, unwrap=unwrap)
