import json
import multiprocessing
import os
from pathlib import Path

import pytest

from orze.reporting.notifications import (
    _format_discord,
    _format_slack,
    _format_telegram,
)
from orze.reporting import notifications as notification_module
from orze.service import watchdog
from orze.service import status as service_status
from orze.service.failure_loop import (
    read_failure_state,
    record_failure,
    record_resolution,
)


def _record_in_process(results_dir: str) -> None:
    record_failure(
        Path(results_dir),
        "test-host",
        "systemd_start_failed",
        ("systemd",),
        now=100.0,
    )


def _state_path(results: Path) -> Path:
    paths = list(results.glob(".orze_watchdog_failures_*.json"))
    assert len(paths) == 1
    return paths[0]


def test_first_repeat_alerts_once_then_obeys_cooldown(tmp_path):
    first = record_failure(
        tmp_path, "host", "systemd_start_failed", ("systemd",), now=10
    )
    assert first["consecutive_count"] == 1
    assert first["alert_due"] is False

    second = record_failure(
        tmp_path, "host", "systemd_start_failed", ("systemd",), now=11
    )
    assert second["consecutive_count"] == 2
    assert second["alert_due"] is True
    assert second["alert_count"] == 1

    third = record_failure(
        tmp_path, "host", "systemd_start_failed", ("systemd",), now=12
    )
    assert third["consecutive_count"] == 3
    assert third["alert_due"] is False

    later = record_failure(
        tmp_path,
        "host",
        "systemd_start_failed",
        ("systemd",),
        now=12 + 6 * 3600,
    )
    assert later["alert_due"] is True
    assert later["alert_count"] == 2


def test_changed_identity_and_resolution_reset_consecutive_count(tmp_path):
    record_failure(
        tmp_path, "host", "runtime_contract_rejected", ("error_a",), now=10
    )
    changed = record_failure(
        tmp_path, "host", "runtime_contract_rejected", ("error_b",), now=11
    )
    assert changed["consecutive_count"] == 1
    assert changed["alert_due"] is False

    assert record_resolution(tmp_path, "host", "service_healthy", now=12)
    resolved = json.loads(_state_path(tmp_path).read_text(encoding="utf-8"))
    assert resolved["active"] is False
    assert resolved["resolution_code"] == "service_healthy"
    again = record_failure(
        tmp_path, "host", "runtime_contract_rejected", ("error_b",), now=13
    )
    assert again["consecutive_count"] == 1


def test_invalid_cooldown_policy_cannot_disable_default_escalation(tmp_path):
    first = record_failure(
        tmp_path,
        "host",
        "systemd_start_failed",
        now=10,
        alert_cooldown_seconds=float("nan"),
    )
    second = record_failure(
        tmp_path,
        "host",
        "systemd_start_failed",
        now=11,
        alert_cooldown_seconds=float("nan"),
    )
    assert first["alert_due"] is False
    assert second["alert_due"] is True


def test_state_is_mode_0600_and_retains_no_raw_identity(tmp_path):
    secret = "token=super-secret-value /private/project/path"
    event = record_failure(
        tmp_path,
        "host/name",
        "runtime_contract_rejected",
        ("systemd", secret),
        now=10,
    )
    state_path = _state_path(tmp_path)
    raw = state_path.read_text(encoding="utf-8")
    assert secret not in raw
    assert "private/project" not in raw
    assert event["fingerprint"] in raw
    assert os.stat(state_path).st_mode & 0o777 == 0o600
    operator_view = read_failure_state(tmp_path, "host/name")
    assert operator_view["valid"] is True
    assert operator_view["failure_code"] == "runtime_contract_rejected"
    assert "identity_parts" not in operator_view


@pytest.mark.parametrize(
    "invalid",
    [
        "{broken",
        "x" * (65 * 1024),
        json.dumps({
            "schema_version": 1,
            "active": True,
            "failure_code": "systemd_start_failed",
            "fingerprint": "0" * 64,
            "consecutive_count": "many",
            "first_seen_epoch": 1,
            "last_seen_epoch": 2,
            "last_alert_epoch": None,
            "alert_count": 0,
        }),
    ],
)
def test_invalid_or_oversized_state_is_recovered_without_using_content(
    tmp_path, invalid
):
    state_path = tmp_path / ".orze_watchdog_failures_host.json"
    state_path.write_text(invalid, encoding="utf-8")
    event = record_failure(
        tmp_path, "host", "systemd_start_failed", ("systemd",), now=10
    )
    assert event["consecutive_count"] == 1
    assert event["recovered_from_invalid_state"] is True
    assert invalid not in state_path.read_text(encoding="utf-8")


def test_state_symlink_is_never_followed(tmp_path):
    target = tmp_path / "target"
    target.write_text("do-not-touch", encoding="utf-8")
    state_path = tmp_path / ".orze_watchdog_failures_host.json"
    state_path.symlink_to(target)
    with pytest.raises(OSError):
        record_failure(
            tmp_path, "host", "systemd_start_failed", ("systemd",), now=10
        )
    assert target.read_text(encoding="utf-8") == "do-not-touch"


def test_nonregular_state_cannot_block_accounting(tmp_path):
    state_path = tmp_path / ".orze_watchdog_failures_host.json"
    os.mkfifo(state_path)
    event = record_failure(
        tmp_path, "host", "systemd_start_failed", ("systemd",), now=10
    )
    assert event["recovered_from_invalid_state"] is True
    assert state_path.is_file()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_unsafe_lock_is_rejected_without_changing_target(tmp_path, link_kind):
    target = tmp_path / "lock-target"
    target.write_text("do-not-touch", encoding="utf-8")
    target.chmod(0o640)
    lock_path = tmp_path / ".orze_watchdog_failures_host.lock"
    if link_kind == "symlink":
        lock_path.symlink_to(target)
    else:
        os.link(target, lock_path)
    with pytest.raises(OSError):
        record_failure(
            tmp_path, "host", "systemd_start_failed", ("systemd",), now=10
        )
    assert target.read_text(encoding="utf-8") == "do-not-touch"
    assert os.stat(target).st_mode & 0o777 == 0o640


def test_unknown_state_fields_are_not_carried_forward(tmp_path):
    record_failure(tmp_path, "host", "systemd_start_failed", now=10)
    state_path = _state_path(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["raw_detail"] = "token=must-not-survive"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    recovered = record_failure(
        tmp_path, "host", "systemd_start_failed", now=11
    )
    raw = state_path.read_text(encoding="utf-8")
    assert recovered["consecutive_count"] == 1
    assert recovered["recovered_from_invalid_state"] is True
    assert "must-not-survive" not in raw


def test_operator_view_reports_invalid_state_without_echoing_it(tmp_path):
    state_path = tmp_path / ".orze_watchdog_failures_host.json"
    state_path.write_text("token=must-not-echo", encoding="utf-8")
    view = read_failure_state(tmp_path, "host")
    assert view == {
        "schema_version": 1,
        "valid": False,
        "error_code": "watchdog_failure_state_invalid",
    }
    assert "must-not-echo" not in json.dumps(view)


def test_service_status_surfaces_only_safe_active_loop_summary(
    tmp_path, monkeypatch, capsys
):
    results = tmp_path / "results"
    results.mkdir()
    for now in (10, 11):
        record_failure(
            results,
            "status-host",
            "runtime_contract_rejected",
            ("token=must-not-print",),
            now=now,
        )
    monkeypatch.setattr(
        service_status,
        "load_service_config",
        lambda: {
            "method": "systemd",
            "results_dir": str(results),
            "stall_threshold": 60,
        },
    )
    monkeypatch.setattr(
        service_status.socket, "gethostname", lambda: "status-host"
    )
    monkeypatch.setattr(service_status, "_is_systemd_active", lambda: False)
    monkeypatch.setattr(
        service_status, "_is_systemd_timer_active", lambda: True
    )
    monkeypatch.setattr(service_status, "_read_pid", lambda *_: None)
    monkeypatch.setattr(
        service_status, "_is_heartbeat_stale", lambda *_: (False, 0)
    )
    monkeypatch.setattr(
        "orze.service.runtime_contract.audit_runtime_contract",
        lambda _: {"contract_ok": False, "errors": ["runtime_drift"]},
    )

    service_status.show_status()
    output = capsys.readouterr().out
    assert "Active: runtime_contract_rejected (consecutive 2)" in output
    assert "token=must-not-print" not in output


@pytest.mark.parametrize("bad_now", [-1, float("nan"), float("inf")])
def test_invalid_framework_clock_is_rejected(tmp_path, bad_now):
    with pytest.raises(ValueError):
        record_failure(
            tmp_path,
            "host",
            "systemd_start_failed",
            ("systemd",),
            now=bad_now,
        )
    assert not list(tmp_path.glob(".orze_watchdog_failures_*.json"))


def test_process_concurrency_loses_no_failures(tmp_path):
    ctx = multiprocessing.get_context("fork")
    processes = [
        ctx.Process(target=_record_in_process, args=(str(tmp_path),))
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    state = json.loads(_state_path(tmp_path).read_text(encoding="utf-8"))
    assert state["consecutive_count"] == 8


def test_watchdog_escalates_repeated_failure_without_logging_raw_output(
    tmp_path, monkeypatch
):
    results = tmp_path / "results"
    results.mkdir()
    log_file = tmp_path / "watchdog.log"
    notifications = []
    monkeypatch.setattr(watchdog.socket, "gethostname", lambda: "test-host")
    monkeypatch.setattr(watchdog, "_read_pid", lambda *_: None)
    monkeypatch.setattr(watchdog, "_is_orze_running", lambda: False)
    monkeypatch.setattr(
        watchdog,
        "_launch_orze",
        lambda _: (_ for _ in ()).throw(
            watchdog.WatchdogLaunchError(
                "runtime_contract_rejected",
                ("systemd_pythonpath_override", "token=must-not-leak"),
            )
        ),
    )
    monkeypatch.setattr(
        watchdog,
        "_notify_failure_loop",
        lambda _cfg, event: notifications.append(dict(event)),
    )
    cfg = {
        "method": "systemd",
        "results_dir": str(results),
        "log_file": str(log_file),
    }

    for _ in range(3):
        with pytest.raises(
            watchdog.WatchdogLaunchError, match="runtime_contract_rejected"
        ):
            watchdog.check_and_restart(cfg)

    log = log_file.read_text(encoding="utf-8")
    state = _state_path(results).read_text(encoding="utf-8")
    assert log.count("ALERT repeated watchdog launch failure") == 1
    assert len(notifications) == 1
    assert notifications[0]["consecutive_count"] == 2
    assert "must-not-leak" not in log
    assert "must-not-leak" not in state


def test_launch_error_prints_only_explicit_closed_vocabulary_details():
    hidden = watchdog.WatchdogLaunchError(
        "systemd_start_failed", ("supersecrettoken",)
    )
    assert "supersecrettoken" not in str(hidden)
    visible = watchdog.WatchdogLaunchError(
        "runtime_contract_rejected",
        ("systemd_restart_policy_drift",),
        display_parts=("systemd_restart_policy_drift",),
    )
    assert "systemd_restart_policy_drift" in str(visible)
    invalid_code = watchdog.WatchdogLaunchError("token=must-not-print")
    assert invalid_code.code == "unclassified_failure"
    assert "must-not-print" not in str(invalid_code)


def test_notification_formatters_use_only_safe_summary_fields():
    data = {
        "host": "node-1",
        "failure_code": "systemd_start_failed",
        "consecutive_count": 2,
        "fingerprint": "0123456789abcdef",
        "raw_detail": "token=must-not-leak",
    }
    slack = json.dumps(_format_slack("watchdog_failure_loop", data))
    discord = json.dumps(_format_discord("watchdog_failure_loop", data))
    _, telegram_payload = _format_telegram(
        "watchdog_failure_loop",
        data,
        {"bot_token": "bot", "chat_id": "chat"},
    )
    telegram = json.dumps(telegram_payload)
    for rendered in (slack, discord, telegram):
        assert "systemd_start_failed" in rendered
        assert "0123456789ab" in rendered
        assert "must-not-leak" not in rendered


def test_failure_loop_event_bypasses_optional_event_filters_safely(monkeypatch):
    sent = []

    def fake_send(url, payload, headers=None, timeout=10):
        sent.append((url, payload, headers))
        return True, None

    monkeypatch.setattr(notification_module, "_notify_send", fake_send)
    cfg = {
        "notifications": {
            "enabled": True,
            "on": ["completed"],
            "channels": [
                {"type": "slack", "webhook_url": "https://slack.invalid"},
                {"type": "discord", "webhook_url": "https://discord.invalid"},
                {"type": "telegram", "bot_token": "bot", "chat_id": "chat"},
                {"type": "webhook", "url": "https://webhook.invalid"},
            ],
        }
    }
    notification_module.notify(
        "watchdog_failure_loop",
        {
            "host": "node-1",
            "failure_code": "systemd_start_failed",
            "consecutive_count": 2,
            "fingerprint": "0123456789ab",
            "raw_detail": "token=must-not-leak",
        },
        cfg,
    )
    assert len(sent) == 4
    rendered = json.dumps(sent)
    assert "systemd_start_failed" in rendered
    assert "must-not-leak" not in rendered
