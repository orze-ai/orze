"""Real parser/loader dispatch with explicit handoff transport and OS doubles.

These are new CLI contracts, not real stop, pidfd, SQLite handoff or successor
proof. The fake CompletedControllerHandoff tests exact class consumption only;
independent product tests must establish the actual operation's qualification.
"""
import copy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from orze import cli
from orze.core.controller_profile import profile_fingerprint
from test_controller_profile import supported


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    import orze.extensions
    import orze.lifecycle
    import orze.engine.orchestrator as orchestrator
    import orze.core.control_outcome as control

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ORZE_CONTROLLER_HANDOFF_FD", raising=False)
    cfg = supported(tmp_path)
    cfg["controller_control"] = {"version": 2, "profile": "local_handoff_v1"}
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    module = ModuleType("orze.engine.controller_handoff")
    module.CompletedControllerHandoff = type("CompletedControllerHandoff", (), {})
    module.restart_controller = Mock(return_value=module.CompletedControllerHandoff())
    module.prepare_successor_entry = Mock(return_value=None)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(orze.extensions, "_find_pro_key", lambda: "test-only-key")
    monkeypatch.setattr(cli, "setup_logging", lambda *args: None)
    monkeypatch.setattr(cli, "_require_controller_runtime", Mock())
    forbidden = []
    for owner, name in ((cli, "detect_all_gpus"), (cli, "maybe_star"),
                        (cli, "stop_running_instance"), (orze.lifecycle, "do_start"),
                        (orze.lifecycle, "do_restart")):
        spy = Mock(side_effect=AssertionError("unexpected legacy/control side effect"))
        monkeypatch.setattr(owner, name, spy)
        forbidden.append(spy)
    run = Mock()
    constructor = Mock(return_value=SimpleNamespace(run=run))
    monkeypatch.setattr(orchestrator, "Orze", constructor)
    sentinel = Mock()
    monkeypatch.setattr(control, "require_controller_start_allowed", sentinel)
    yield SimpleNamespace(module=module, cfg=cfg, path=path, constructor=constructor,
                          run=run, sentinel=sentinel, forbidden=forbidden)
    for spy in forbidden:
        spy.assert_not_called()


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["orze", *args])
    return cli.main()


@pytest.mark.parametrize("args", [
    ["restart", "--request-id", "repeat.1:stable", "--timeout", "7"],
    ["--restart", "--request-id", "repeat.1:stable", "--timeout", "7"],
    ["--request-id", "repeat.1:stable", "restart", "--timeout", "7"],
])
def test_both_restart_routes_pass_exact_request_without_mutating_training_budget(handoff, monkeypatch, args):
    c = handoff
    assert invoke(monkeypatch, *args) == 0
    assert c.module.restart_controller.call_count == 1
    loaded = c.module.restart_controller.call_args.args[0]
    assert loaded["timeout"] == 3600
    assert loaded["_controller_profile_fingerprint"] == profile_fingerprint(loaded)
    assert c.module.restart_controller.call_args.kwargs == {"request_id": "repeat.1:stable", "timeout": 7}
    c.module.prepare_successor_entry.assert_not_called()
    c.constructor.assert_not_called()
    c.sentinel.assert_not_called()


@pytest.mark.parametrize("request_id", [None, "", "unsafe/key", "x" * 129])
def test_missing_or_invalid_stable_key_never_invokes_handoff(handoff, monkeypatch, request_id):
    args = ["--restart"] + ([] if request_id is None else ["--request-id", request_id])
    assert invoke(monkeypatch, *args) == 75
    handoff.module.restart_controller.assert_not_called()
    handoff.constructor.assert_not_called()


@pytest.mark.parametrize("kind", ["true", "label", "subclass", "exception", "missing-module"])
def test_unqualified_results_and_failures_never_return_success(handoff, monkeypatch, kind):
    c = handoff
    if kind == "true":
        c.module.restart_controller.return_value = True
    elif kind == "label":
        c.module.restart_controller.return_value = SimpleNamespace(status="confirmed")
    elif kind == "subclass":
        c.module.restart_controller.return_value = type("Other", (c.module.CompletedControllerHandoff,), {})()
    elif kind == "exception":
        c.module.restart_controller.side_effect = OSError("synthetic operation uncertainty")
    else:
        monkeypatch.setitem(sys.modules, c.module.__name__, None)
    assert invoke(monkeypatch, "restart", "--request-id", "one") == 75
    c.constructor.assert_not_called()
    c.sentinel.assert_not_called()


@pytest.mark.parametrize("args", [["restart"], ["--restart"]])
def test_v1_stays_stop_only_and_never_calls_v2_service(handoff, monkeypatch, args):
    import orze.lifecycle
    from orze.core.control_outcome import StopOutcome
    c = handoff
    cfg = copy.deepcopy(c.cfg)
    cfg["controller_control"] = {"version": 1, "profile": "local_stop_v1"}
    c.path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(orze.lifecycle, "do_restart", Mock(return_value=StopOutcome("requested", "stop_requested")))
    monkeypatch.setattr(cli, "stop_running_instance", Mock(return_value=True))
    assert invoke(monkeypatch, *args, "--request-id", "not-an-upgrade") == 75
    c.module.restart_controller.assert_not_called()
    c.constructor.assert_not_called()


@pytest.mark.parametrize("args", [["start"], ["--enable"],
    ["resume", "idea", "--from", "interruption.json"], ["restart", "--foreground"]])
def test_v2_does_not_grant_other_start_or_resume_modes(handoff, monkeypatch, args):
    assert invoke(monkeypatch, *args) == 2
    handoff.module.restart_controller.assert_not_called()
    handoff.constructor.assert_not_called()
    handoff.sentinel.assert_not_called()


@pytest.mark.parametrize("raw,args", [("", []), ("1048577", []),
    ("3", ["--stop"]), ("3", ["init", "project"]), ("3", ["--report-only"])])
def test_claimed_private_fd_cannot_escape_through_early_cli_modes(handoff, monkeypatch, raw, args):
    monkeypatch.setenv("ORZE_CONTROLLER_HANDOFF_FD", raw)
    load = Mock(side_effect=AssertionError("invalid private entry must precede configuration IO"))
    monkeypatch.setattr(cli, "load_project_config", load)
    assert invoke(monkeypatch, *args) == 75
    load.assert_not_called()
    handoff.module.prepare_successor_entry.assert_not_called()
    handoff.constructor.assert_not_called()


@pytest.mark.parametrize("reject", [False, True])
def test_private_entry_is_checked_before_sentinel_gpu_or_constructor(handoff, monkeypatch, reject):
    c = handoff
    order = []
    monkeypatch.setenv("ORZE_CONTROLLER_HANDOFF_FD", "3")
    monkeypatch.setattr(cli, "_require_controller_runtime", lambda cfg: order.append("runtime"))
    def prepare(cfg, args):
        order.append("entry")
        assert cfg["_controller_profile_fingerprint"] == profile_fingerprint(cfg)
        assert args.command is None
        if reject:
            raise ValueError("synthetic private channel refusal")
    c.module.prepare_successor_entry.side_effect = prepare
    c.sentinel.side_effect = lambda root: order.append("sentinel")
    c.constructor.side_effect = lambda *args, **kwargs: (order.append("construct") or SimpleNamespace(run=c.run))
    assert invoke(monkeypatch, "--no-admin") == (75 if reject else None)
    assert order == (["runtime", "entry"] if reject else ["runtime", "entry", "sentinel", "construct"])
    assert c.run.call_count == (0 if reject else 1)


def test_ordinary_v2_foreground_has_no_private_entry_or_legacy_bootstrap(handoff, monkeypatch):
    import threading
    monkeypatch.setattr(threading, "Thread", Mock(side_effect=AssertionError("no implicit admin")))
    assert invoke(monkeypatch) is None
    handoff.module.prepare_successor_entry.assert_not_called()
    assert handoff.constructor.call_args.args[0] == [2, 4]
    handoff.run.assert_called_once()


def test_replication_request_argument_keeps_its_existing_destination(monkeypatch, handoff):
    import orze.cli_replication
    service = Mock(return_value=0)
    monkeypatch.setattr(orze.cli_replication, "run_replication", service)
    assert invoke(monkeypatch, "replicate", "source", "--request-id", "replica-key") == 0
    args = service.call_args.args[0]
    assert args.request_id == "replica-key"
    assert args.controller_request_id is None
    handoff.module.restart_controller.assert_not_called()
