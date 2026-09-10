"""A native completion token must survive its actual downstream consumers.

Real Lake, native launch/completion/retry, source qualification and scheduling
are exercised. GPU/process boundaries are synthetic. An empty ideas mapping
excludes the legitimate independent backlog scheduler from event-only cases.
"""
import json
import threading
from contextlib import nullcontext
from types import SimpleNamespace

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator, launcher, orchestrator, phases
from orze.engine.evaluation_retry import request_evaluation_retry
from orze.engine.failure import _reset_idea_for_retry
from orze.engine.scheduler import claim
from orze.reporting.leaderboard import NotificationProcessor

from test_native_training_caller_boundaries import case as training_case, _launch as train, Child
from test_stale_evaluation_completion import project, _prepare, _launch, _exit, _lifecycle


def controller(case):
    return SimpleNamespace(cfg=case.cfg, results_dir=case.results, lake=case.lake,
                           gpu_ids=[0], active={}, active_evals={}, pending_evals=[])


def rotate_training(case, monkeypatch):
    old = train(case)
    old.process.returncode = 1
    old.process.wait = lambda timeout=None: 1
    events = launcher.check_active({0: old}, case.results, case.cfg, {}, lake=case.lake)
    assert events == [(case.idea, 0)]
    assert events[0].attempt_ref == old.attempt_ref
    _reset_idea_for_retry(case.folder, release_claim=True, lake=case.lake)
    assert case.lake.record_state_transition(case.idea, "FAILED", "QUEUED", "explicit test retry")
    assert claim(case.idea, case.results, 0, lake=case.lake)
    # Popen itself is fake, but each native launch still gets a fresh claim,
    # generic generation, intent and start receipt through production APIs.
    next_pid = old.process.pid + 1
    def new_process(*args, **kwargs):
        nonlocal next_pid
        case.popen_calls.append(True)
        case.child = Child(pid=next_pid)
        next_pid += 1
        return case.child
    monkeypatch.setattr(launcher.subprocess, "Popen", new_process)
    current = train(case)
    assert current.attempt_ref.generation > old.attempt_ref.generation
    assert old.process.returncode == 1 and current.process is not old.process
    return events[0], current


def test_delayed_training_event_cannot_deliver_the_running_retry(training_case, monkeypatch):
    case = training_case
    event, current = rotate_training(case, monkeypatch)
    metrics = case.folder / "metrics.json"
    metrics.write_text('{"status":"IN_PROGRESS","step":3}')
    history = _lifecycle(case.lake)
    c = controller(case)

    delivered, backlog = phases.OrzePhaseMixin._launch_evals(c, [event], [], {})

    assert delivered == [], "A's accepted terminal cannot report running B as finished"
    assert backlog == []
    assert _lifecycle(case.lake) == history
    assert current_attempt(case.lake.conn, case.idea, "training")["state"] == "RUNNING"
    assert current.process.returncode is None


def test_delayed_training_event_cannot_start_eval_from_unaccepted_retry_output(training_case, monkeypatch):
    case = training_case
    event, current = rotate_training(case, monkeypatch)
    case.cfg.update(eval_script="unused-evaluation.py", eval_output="assessment.json")
    # A trainer may write its artifact just before exit. B has NOT delivered
    # a terminal yet, so A's token cannot authorize B's downstream evaluator.
    (case.folder / "metrics.json").write_text('{"status":"COMPLETED","score":0}')
    monkeypatch.setattr(evaluator, "gpu_execution_lease", lambda *a, **k: nullcontext(()))
    monkeypatch.setattr(evaluator, "_verify_gpu_free", lambda *a, **k: None)
    c = controller(case)
    history = _lifecycle(case.lake)
    popen_count = len(case.popen_calls)

    delivered, backlog = phases.OrzePhaseMixin._launch_evals(c, [event], [], {})

    assert len(case.popen_calls) == popen_count, "stale training delivery launched an evaluator"
    assert delivered == [] and backlog == [] and c.active_evals == {}
    assert _lifecycle(case.lake) == history
    assert current_attempt(case.lake.conn, case.idea, "evaluation") is None
    assert current.process.returncode is None


def test_already_accepted_native_eval_none_is_not_a_second_finished_event(project):
    p = project
    folder = _prepare(p, "idea-native-none")
    ep = _launch(p, folder.name)
    _exit(p, ep, 0)
    events = evaluator.check_active_evals({0: ep}, p.results, p.cfg, lake=p.lake)
    assert events == [(folder.name, 0)] and events[0].attempt_ref == ep.attempt_ref
    before = _lifecycle(p.lake)
    c = controller(p)

    # The real launch_eval returns None for an accepted native terminal. The
    # phase must not reinterpret that absence as a newly accepted completion.
    delivered, backlog = phases.OrzePhaseMixin._launch_evals(c, [(folder.name, 0)], [], {})

    assert delivered == [], "finish_without_process revived an already delivered native terminal"
    assert backlog == [] and c.active_evals == {}
    assert _lifecycle(p.lake) == before
    assert p.popen.call_count == 1


def test_notification_counter_cannot_attribute_current_result_to_old_eval_event(project):
    p = project
    folder = _prepare(p, "idea-late-notification")
    old = _launch(p, folder.name)
    _exit(p, old, 1)
    old_events = evaluator.check_active_evals({0: old}, p.results, p.cfg, lake=p.lake)
    assert old_events[0].attempt_ref == old.attempt_ref
    assert request_evaluation_retry(folder.name, p.results, p.cfg, p.lake)["status"] == "evaluation_retry_pending"
    current = _launch(p, folder.name)
    _exit(p, current, 0)
    assert evaluator.check_active_evals({0: current}, p.results, p.cfg, lake=p.lake) == [(folder.name, 0)]
    reporter = NotificationProcessor(p.results, p.cfg, p.lake)
    candidates = [{"id": folder.name}]
    # Real current qualification establishes B as the baseline independently
    # of any finished batch. No mock metric or fake qualification is involved.
    reporter.process([], candidates, {}, {}, 0, lambda *a: None, lambda: [])
    assert reporter.get_state()["best_idea_id"] == folder.name
    before = reporter.get_state()

    reporter.process(old_events, candidates, {}, {}, 0, lambda *a: None, lambda: [])

    assert reporter.get_state() == before, "A's obsolete event incremented B's completion counter"


def test_main_loop_does_not_launch_post_script_from_eval_event_invalidated_before_delivery(project, monkeypatch):
    p = project
    folder = _prepare(p, "idea-delayed-post")
    old = _launch(p, folder.name)
    _exit(p, old, 1)
    p.cfg.update(_managed_idea_id=folder.name, ideas_file=str(p.results / "unused.md"),
                 timeout=60, poll=1, roles={}, notifications={"enabled": False},
                 post_scripts=[{"script": "unused-post-script.py", "name": "post-step"}])
    runner = orchestrator.Orze.__new__(orchestrator.Orze)
    runner.cfg, runner.results_dir, runner.lake = p.cfg, p.results, p.lake
    runner.gpu_ids, runner.active, runner.active_evals = [0], {}, {0: old}
    runner.active_roles, runner.pending_evals = {}, []
    runner.running, runner.iteration, runner.once = True, 0, False
    runner._auto_gpu_mode, runner._leader_handle = False, None
    runner._stop_event = threading.Event()
    runner._check_stop_all = lambda: False
    runner._check_disabled = lambda: False
    runner._remove_pid_file = lambda: None
    # B is fake-running; no OS child actually exists to shut down in teardown.
    runner._graceful_shutdown = lambda **kwargs: None
    monkeypatch.setattr(orchestrator, "check_disk_space", lambda *a: True)
    monkeypatch.setattr("orze.engine.health.HealthMonitor", lambda *a: SimpleNamespace(
        check_before_write=lambda: True, retry_delay=0))
    monkeypatch.setattr("orze.extensions.has_pro", lambda: False)
    actual_check = evaluator.check_active_evals
    observed = []

    def finish_then_retry(active, results, cfg, lake=None):
        events = actual_check(active, results, cfg, lake=lake)
        assert events == [(folder.name, 0)]
        assert request_evaluation_retry(folder.name, results, cfg, lake)["status"] == "evaluation_retry_pending"
        current = evaluator.launch_eval(folder.name, 0, results, cfg, lake=lake)
        assert current is not None and current.attempt_id != old.attempt_id
        active[0] = current
        observed.append((events[0], current, p.popen.call_count))
        runner.running = False  # Stop after this real post-eval consumer block.
        return events

    monkeypatch.setattr(orchestrator, "check_active_evals", finish_then_retry)
    actual_popen = p.popen.side_effect
    def process_boundary(cmd, **kwargs):
        process = actual_popen(cmd, **kwargs)
        if "unused-post-script.py" in cmd:
            process.returncode = 0
        return process
    p.popen.side_effect = process_boundary

    runner._run_leased()

    assert len(observed) == 1, "exercise actual evaluator completion then real retry"
    event, current, before_post = observed[0]
    assert event.attempt_ref == old.attempt_ref
    assert p.popen.call_count == before_post, "main loop dropped A's token and launched a post-script during B"
    assert not (folder / "post-step.log").exists()
    assert current.process.returncode is None
