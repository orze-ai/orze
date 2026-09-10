"""Additional actual --once/auto-GPU consumers; new mechanism acceptance."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from orze.engine import evaluator, orchestrator
from orze.engine.evaluation_retry import request_evaluation_retry

from test_completion_event_consumers import rotate_training
from test_native_training_caller_boundaries import case as training_case
from test_stale_evaluation_completion import project, _prepare, _launch, _exit


def runner_for(p, idea, monkeypatch):
    p.cfg.update(_managed_idea_id=idea, ideas_file=str(p.results / "unused.md"),
                 results_dir=str(p.results), timeout=60, poll=1, roles={},
                 notifications={"enabled": False})
    runner = orchestrator.Orze.__new__(orchestrator.Orze)
    runner.cfg, runner.results_dir, runner.lake = p.cfg, p.results, p.lake
    runner.gpu_ids, runner.active, runner.active_evals = [0], {}, {}
    runner.active_roles, runner.pending_evals = {}, []
    runner.failure_counts, runner.fix_counts = {}, {}
    runner.running, runner.iteration, runner.once = True, 0, True
    runner._auto_gpu_mode, runner._leader_handle = False, None
    runner._stop_event = threading.Event()
    runner.slot_mgr = SimpleNamespace(mode="exclusive", max_jobs_per_gpu=1)
    runner._check_stop_all = lambda: False
    runner._check_disabled = lambda: False
    runner._remove_pid_file = lambda: None
    runner._graceful_shutdown = lambda **kwargs: None
    runner._sync_managed_idea = lambda *args: ({}, [], set(), {})
    runner._launch_training = lambda *args: [0]
    runner._capture_campaign_efficiency_evidence = lambda *args: None
    runner._capture_campaign_progress_evidence = lambda *args: None
    runner._capture_managed_wait_campaign_evidence = lambda *args: True
    monkeypatch.setattr(orchestrator, "check_disk_space", lambda *a: True)
    monkeypatch.setattr("orze.engine.health.HealthMonitor", lambda *a: SimpleNamespace(
        check_before_write=lambda: True, retry_delay=0))
    monkeypatch.setattr("orze.extensions.has_pro", lambda: False)
    monkeypatch.setattr(orchestrator.time, "sleep", lambda *a: None)
    return runner


def test_once_eval_wait_preserves_event_identity_after_real_retry(project, monkeypatch):
    p = project
    folder = _prepare(p, "idea-once-delivery")
    old = _launch(p, folder.name)
    _exit(p, old, 1)
    p.cfg["post_scripts"] = [{"script": "unused-once-post.py", "name": "once-post"}]
    runner = runner_for(p, folder.name, monkeypatch)
    def register_for_wait(*args):
        runner.active_evals[0] = old
        return [0]
    runner._launch_training = register_for_wait
    actual = evaluator.check_active_evals
    observed = []
    def complete_then_retry(active, results, cfg, lake=None):
        events = actual(active, results, cfg, lake=lake)
        assert events == [(folder.name, 0)]
        assert request_evaluation_retry(folder.name, results, cfg, lake)["status"] == "evaluation_retry_pending"
        current = evaluator.launch_eval(folder.name, 0, results, cfg, lake=lake)
        assert current is not None
        active[0] = current
        observed.append((events[0], current, p.popen.call_count))
        runner.running = False
        return events
    monkeypatch.setattr(orchestrator, "check_active_evals", complete_then_retry)

    runner._run_leased()

    assert len(observed) == 1
    event, current, before_post = observed[0]
    assert event.attempt_ref == old.attempt_ref
    assert p.popen.call_count == before_post
    assert not (folder / "once-post.log").exists()
    assert current.process.returncode is None


def test_auto_gpu_mode_cannot_use_new_generation_metrics_for_an_old_event(training_case, monkeypatch):
    p = training_case
    event, current = rotate_training(p, monkeypatch)
    (p.folder / "metrics.json").write_text('{"status":"COMPLETED","training_time":99}')
    runner = runner_for(p, p.idea, monkeypatch)
    runner.active = {0: current}
    runner._auto_gpu_mode = True
    delivered = []
    def delayed_delivery(*args, **kwargs):
        delivered.append(event)
        runner.running = False
        return [event]  # Previously accepted real completion, delayed in transit.
    monkeypatch.setattr(orchestrator, "check_active", delayed_delivery)
    usage = Mock(return_value={0: (10, 1000)})
    monkeypatch.setattr("orze.engine.gpu_slots._query_all_gpu_usage", usage)

    runner._run_leased()

    assert delivered == [event]
    usage.assert_not_called()
    assert runner.slot_mgr.mode == "exclusive"
    assert runner.slot_mgr.max_jobs_per_gpu == 1
