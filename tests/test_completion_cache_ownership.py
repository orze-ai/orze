"""New source-owned cache mechanism, not a claimed success-to-retry old bug.

Current legal COMPLETE transitions do not permit evaluation retry. These tests
preserve the boundary for future revision APIs without fabricating FSM edges.
Caches are diagnostic/dedup data; neither writes nor notifications are ACKs.
"""
import logging

import pytest

from orze.core.integrity import hash_config, load_hashes, save_hash
from orze.engine import evaluator
from orze.engine.attempt_effect_lock import AttemptEffectBusy, attempt_effect_lock
from orze.engine.evaluation_retry import request_evaluation_retry
from orze.reporting import leaderboard

from test_stale_evaluation_completion import project, _prepare, _launch, _exit


def completed(p):
    folder = _prepare(p, "idea-cache")
    ep = _launch(p, folder.name)
    _exit(p, ep, 0)
    event = evaluator.check_active_evals({0: ep}, p.results, p.cfg, lake=p.lake)[0]
    assert p.lake.get_fsm_state(folder.name) == "COMPLETE"
    (folder / "idea_config.yaml").write_text("seed: 17\n")
    return folder, event, leaderboard.NotificationProcessor(p.results, p.cfg, lake=p.lake)


def process(p, reporter, event, callback):
    reporter.process([event], [], {}, p.lake.get_lifecycle_counts(), 0,
                     callback, lambda: [])


def test_native_cache_writes_hold_source_guard_but_notify_and_recovery_do_not(project, monkeypatch):
    p = project
    folder, event, reporter = completed(p)
    seen = []
    original = reporter._recover_overrides

    def recover(*args):
        assert not (folder / "_attempt_effect.lock").exists()
        assert not p.lake.conn.in_transaction
        seen.append("recover")
        return original(*args)

    def notify(*args):
        assert not (folder / "_attempt_effect.lock").exists()
        assert not p.lake.conn.in_transaction
        seen.append("notify")

    def save(idea_id, overrides):
        assert (folder / "_attempt_effect.lock").exists()
        assert not p.lake.conn.in_transaction
        # The actual cache callback is the contender seam: retry checks the
        # common effect guard before even evaluating its terminal-state rule.
        with pytest.raises(AttemptEffectBusy):
            request_evaluation_retry(idea_id, p.results, p.cfg, p.lake)
        save_hash(p.results, idea_id, overrides)
        seen.append("save")

    monkeypatch.setattr(reporter, "_recover_overrides", recover)
    monkeypatch.setattr(leaderboard, "notify", notify)
    process(p, reporter, event, save)

    assert seen == ["notify", "recover", "save"]
    assert load_hashes(p.results) == {hash_config({"seed": 17}): folder.name}
    assert p.lake.get(folder.name)["eval_metrics"]["quality"] == 0
    assert not (folder / "_attempt_effect.lock").exists()


def test_competing_effect_owner_prevents_both_native_cache_callbacks(project, monkeypatch):
    p = project
    folder, event, reporter = completed(p)
    monkeypatch.setattr(leaderboard, "notify", lambda *a: None)
    before = p.lake.get(folder.name)["eval_metrics"]
    saved = []

    with attempt_effect_lock(folder):
        process(p, reporter, event, lambda *args: saved.append(args))

    assert p.lake.get(folder.name)["eval_metrics"] == before
    assert saved == []
    assert load_hashes(p.results) == {}


def test_confirmation_revoked_during_transport_prevents_late_native_cache_write(project, monkeypatch):
    p = project
    folder, event, reporter = completed(p)
    before = p.lake.get(folder.name)["eval_metrics"]
    revoked = []
    saved = []

    def notify(*args):
        if not revoked:
            (folder / "_execution_effects" / event.attempt_ref.attempt_id / "committed.json").unlink()
            revoked.append(True)

    monkeypatch.setattr(leaderboard, "notify", notify)
    process(p, reporter, event, lambda *args: saved.append(args))

    assert revoked == [True]
    assert p.lake.get(folder.name)["eval_metrics"] == before
    assert saved == []


def test_cache_callback_failure_is_diagnostic_not_an_ack(project, monkeypatch, caplog):
    p = project
    folder, event, reporter = completed(p)
    monkeypatch.setattr(leaderboard, "notify", lambda *a: None)
    calls = []

    def save(*args):
        calls.append(args)
        raise OSError("simulated cache storage unavailable")

    with caplog.at_level(logging.DEBUG, logger="orze"):
        process(p, reporter, event, save)

    assert len(calls) == 1
    assert load_hashes(p.results) == {}
    assert "Completion cache update unavailable" in caplog.text
    assert not (folder / "_attempt_effect.lock").exists()
    assert p.lake.get_fsm_state(folder.name) == "COMPLETE"
