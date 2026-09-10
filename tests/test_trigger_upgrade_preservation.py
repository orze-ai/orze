"""Native upgrade is not authority to discard an outstanding trigger."""
import json
from unittest.mock import Mock

from orze.engine import lifecycle, upgrade


def _upgrade_fixture(results, monkeypatch):
    results.mkdir()
    (results / upgrade.STAMP_FILENAME).write_text(
        json.dumps({"orze": "old", "orze_pro": "old"}), encoding="utf-8")
    current = {"orze": "new", "orze_pro": "new"}
    monkeypatch.setattr(upgrade, "_current_versions", lambda: current)
    trigger = results / "_trigger_worker"
    trigger.write_text("OUTSTANDING OPERATOR REQUEST", encoding="utf-8")
    (results / "_fsm_activity.jsonl").write_text("derived legacy cache", encoding="utf-8")
    return trigger


def test_native_startup_preserves_not_yet_received_trigger_across_upgrade(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    trigger = _upgrade_fixture(results, monkeypatch)
    before = trigger.read_bytes()
    for name in ("fs_startup_check", "cleanup_stale_locks", "reconcile_stale_running",
                 "reconcile_running_dead_pids", "print_startup_summary"):
        monkeypatch.setattr(lifecycle, name, Mock(return_value=True))
    # Actual upgrade and nonce-recovery functions; empty locks need no signals.
    lifecycle.startup_checks(results, {"_orze_dir": str(tmp_path / ".orze")},
                             "test-host", "test-instance")
    assert trigger.exists(), "Package upgrade does not prove task delivery"
    assert trigger.read_bytes() == before
    assert not (results / "_fsm_activity.jsonl").exists()


def test_explicit_preservation_keeps_live_and_archived_inputs(tmp_path, monkeypatch):
    results = tmp_path / "results"
    trigger = _upgrade_fixture(results, monkeypatch)
    archived = results / "_trigger_worker.archived.old"
    archived.write_text("HISTORICAL REQUEST", encoding="utf-8")
    result = upgrade.check_and_clean(results, preserve_triggers=True)
    assert result["upgraded"]
    assert trigger.read_text(encoding="utf-8") == "OUTSTANDING OPERATOR REQUEST"
    assert archived.read_text(encoding="utf-8") == "HISTORICAL REQUEST"
    assert result["cleaned"] == ["_fsm_activity.jsonl"]
    assert upgrade.check_and_clean(results, preserve_triggers=True)["cleaned"] == []
