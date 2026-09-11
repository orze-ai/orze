"""Real foreground CPU action with a controlled health/wait observation seam.

The worker, deadline, budget, launch and cleanup are real. The health result is
injected and Event.wait records its argument then requests loop shutdown: this
does not claim an actual 30-second wait, real filesystem failure or CPU load.
"""
import time

from test_cpu_product_loop import project, task


def test_active_action_health_retry_keeps_deadline_harvest_responsive(project, monkeypatch):
    from orze.engine import cpu_phase, health, native_cpu_action as native

    root, cfg, run = project
    cfg["execution"]["wall_budget_seconds"] = 10
    task(root, outputs={}, program="import time; time.sleep(10)")
    captured, waits = [], []
    real_launch = native.launch
    real_initialize = cpu_phase.initialize

    def launch(*args, **kwargs):
        handle = real_launch(*args, **kwargs)
        captured.append(handle)
        return handle

    def initialize(engine, *args, **kwargs):
        real_initialize(engine, *args, **kwargs)
        underlying = engine._stop_event

        class ObservedWait:
            def is_set(self):
                return underlying.is_set()

            def set(self):
                return underlying.set()

            def wait(self, timeout=None):
                handle = captured[0]
                owner = native._OWNERS[id(handle)]
                waits.append({"requested": timeout, "deadline": owner.deadline,
                              "observed_at": time.monotonic(), "returncode": handle.process.poll()})
                engine.running = False
                underlying.set()
                return True

        engine._stop_event = ObservedWait()

    monkeypatch.setattr(native, "launch", launch)
    monkeypatch.setattr(cpu_phase, "initialize", initialize)
    monkeypatch.setattr(health.HealthMonitor, "check_before_write", lambda self: not captured)

    assert run(once=False) in (None, 0)
    assert len(captured) == len(waits) == 1
    observation = waits[0]
    assert observation["returncode"] is None
    assert observation["deadline"] > observation["observed_at"]
    assert 0 <= observation["requested"] <= 0.05
    closure = captured[0].process.closure_receipt()
    assert closure["wait_proof"] == "ECHILD_WALL"
    assert closure["stop_requested"] is True
