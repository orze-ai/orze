"""Explicit C2a migration boundary for pre-supervisor OS-double fixtures.

Install only in tests that intentionally used fake Popen. This is not an
autouse fixture or production fallback. It preserves their OS-double behavior
while declaring the new supervision identity and closure assumptions. Real CPU
tests explicitly restore the real prepare_supervised entry point.
"""
from copy import deepcopy
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
        return getattr(self._sim_wrapped, name)

    def __setattr__(self, name, value):
        if name.startswith("_sim_"):
            object.__setattr__(self, name, value)
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
