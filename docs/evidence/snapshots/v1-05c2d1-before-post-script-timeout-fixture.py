"""New event/source/action contracts, not old-release behavior red tests."""
import json
from contextlib import nullcontext

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator, launcher, phases
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.completion_events import completion_is_current, require_completion
from orze.engine.evaluation_retry import EvaluationRetryError, request_evaluation_retry
from orze.idea_lake import IdeaLake

from test_completion_event_consumers import controller
from test_native_training_caller_boundaries import case as training_case, _launch as train
from test_stale_evaluation_completion import project, _prepare, _launch, _exit


def accepted(p, name="idea-source"):
    folder = _prepare(p, name)
    ep = _launch(p, name)
    _exit(p, ep, 1)
    events = evaluator.check_active_evals({0: ep}, p.results, p.cfg, lake=p.lake)
    assert len(events) == 1
    return folder, events[0]


def test_event_reference_cannot_be_mutated_through_attributes_or_dict(project):
    _, event = accepted(project)
    assert tuple(event) == (event.idea_id, 0)
    assert event == (event.idea_id, 0) and (event.idea_id, 0) == event
    assert len(event) == 2 and event[0] == event.idea_id
    with pytest.raises(AttributeError):
        event.attempt_ref = None
    with pytest.raises(AttributeError):
        event.__dict__["attempt_ref"] = None
    assert completion_is_current(event, project.lake, project.results)


@pytest.mark.parametrize("wrong_lake", [False, True])
def test_native_completion_requires_its_real_catalog(project, wrong_lake, tmp_path):
    _, event = accepted(project)
    lake = IdeaLake(tmp_path / "different.db") if wrong_lake else None
    try:
        assert not completion_is_current(event, lake, project.results)
        assert not completion_is_current(tuple(event), lake, project.results)
    finally:
        if lake is not None:
            lake.close()


def test_closed_sql_without_confirmation_cannot_authorize_delivery(project):
    folder, event = accepted(project)
    (folder / "_execution_effects" / event.attempt_ref.attempt_id / "committed.json").unlink()
    assert not completion_is_current(event, project.lake, project.results)


@pytest.mark.parametrize("origin", ["pending", "backlog"])
def test_independent_eval_schedule_binds_current_accepted_native_training(training_case, monkeypatch, origin):
    c = training_case
    c.cfg.update(eval_script="unused-evaluator.py", eval_output="assessment.json")
    tp = train(c)
    tp.process.returncode = 0
    tp.process.wait = lambda timeout=None: 0
    (c.folder / "metrics.json").write_text('{"status":"COMPLETED","score":0}')
    (c.folder / "best_model.pt").write_bytes(b"checkpoint")
    event = launcher.check_active({0: tp}, c.results, c.cfg, {}, lake=c.lake)[0]
    monkeypatch.setattr(evaluator, "gpu_execution_lease", lambda *a, **k: nullcontext(()))
    monkeypatch.setattr(evaluator, "_verify_gpu_free", lambda *a, **k: None)
    monkeypatch.setattr(phases, "get_gpu_memory_used", lambda *a: 0)
    monkeypatch.setattr(phases, "_eval_already_running", lambda *a: False)
    from supervision_fixture import install
    install(monkeypatch)
    ctl = controller(c)
    if origin == "pending":
        ctl.pending_evals.append((c.idea, 0))
    delivered, _ = phases.OrzePhaseMixin._launch_evals(
        ctl, [], [], {c.idea: {}} if origin == "backlog" else {})
    assert delivered == []
    assert len(ctl.active_evals) == 1
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    assert row["state"] == "RUNNING"
    assert row["binding"]["source_ref"]["attempt_id"] == tp.attempt_id
    # Evaluation changed lifecycle state after training; historical lifecycle
    # need not equal today's global fence to remain the accepted source.
    assert require_completion(event, c.lake, c.results)["state"] == "TERMINAL"


def test_two_scripts_use_separate_generations_and_repeated_delivery_does_not_restart_first(project):
    p = project
    folder, event = accepted(p)
    p.cfg["post_scripts"] = [{"script": "post-one.py", "name": "one"},
                             {"script": "post-two.py", "name": "two"}]
    actual = p.popen.side_effect
    seen = []
    def process(cmd, **kwargs):
        row = current_attempt(p.lake.conn, folder.name, "post_script")
        assert row["state"] == "LAUNCHING"
        assert not p.lake.conn.in_transaction
        assert not (folder / "_attempt_effect.lock").exists()
        assert row["binding"]["source_ref"]["attempt_id"] == event.attempt_ref.attempt_id
        with pytest.raises((EvaluationRetryError, AttemptEffectBusy)):
            request_evaluation_retry(folder.name, p.results, p.cfg, p.lake)
        seen.append(row["generation"])
        child = actual(cmd, **kwargs)
        child.returncode = 0
        return child
    p.popen.side_effect = process
    evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg, lake=p.lake, source_event=event)
    assert seen == [1, 2]
    assert current_attempt(p.lake.conn, folder.name, "post_script")["state"] == "TERMINAL"
    before = p.popen.call_count
    evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg, lake=p.lake, source_event=event)
    assert p.popen.call_count == before
    assert seen == [1, 2]
    # These action attempts are not evaluations or new source observations.
    assert p.lake.get_fsm_state(folder.name) == "FAILED"
    assert current_attempt(p.lake.conn, folder.name, "evaluation")["attempt_id"] == event.attempt_ref.attempt_id


def test_unknown_post_script_popen_effect_keeps_durable_hold_and_cannot_replay(project):
    p = project
    folder, event = accepted(p)
    p.cfg["post_scripts"] = [{"script": "post-uncertain.py"}]
    p.popen.side_effect = RuntimeError("unknown subprocess handoff")
    with pytest.raises(AttemptEffectInDoubt):
        evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg, lake=p.lake, source_event=event)
    row = current_attempt(p.lake.conn, folder.name, "post_script")
    assert row["state"] == "LAUNCHING"
    before = p.popen.call_count
    with pytest.raises(AttemptEffectBusy):
        evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg, lake=p.lake, source_event=event)
    with pytest.raises((EvaluationRetryError, AttemptEffectBusy)):
        request_evaluation_retry(folder.name, p.results, p.cfg, p.lake)
    assert p.popen.call_count == before
    assert not (folder / "_compute_receipts" / row["attempt_id"] / "terminal.json").exists()
    p.reaper.assert_not_called()


def test_post_script_unconfirmed_stop_keeps_nonterminal_and_never_reaps_twice(project):
    p = project
    folder, event = accepted(p)
    p.cfg["post_scripts"] = [{"script": "post-running.py", "timeout": 0}]
    p.reaper.side_effect = lambda *a, **k: False
    with pytest.raises(AttemptEffectInDoubt):
        evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg, lake=p.lake, source_event=event)
    row = current_attempt(p.lake.conn, folder.name, "post_script")
    assert row["state"] == "RUNNING"
    assert (folder / "_execution_stops" / row["attempt_id"] / "requested.json").exists()
    assert not (folder / "_execution_stops" / row["attempt_id"] / "confirmed.json").exists()
    assert not (folder / "_compute_receipts" / row["attempt_id"] / "terminal.json").exists()
    before = p.popen.call_count
    evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg, lake=p.lake, source_event=event)
    assert p.popen.call_count == before
    p.reaper.assert_called_once()
