"""Exercise the real periodic maintenance branch, with no GPU or daemon IO."""
from types import SimpleNamespace

import pytest

from orze.engine import orchestrator


def test_periodic_cleanup_passes_controller_catalog_to_public_consumer(tmp_path, monkeypatch):
    class MaintenanceReached(Exception):
        pass

    results = tmp_path / "results"
    results.mkdir()
    runner = orchestrator.Orze.__new__(orchestrator.Orze)
    runner.cfg = {"ideas_file": str(tmp_path / "ideas.md"),
                  "results_dir": str(results), "timeout": 60, "poll": 1,
                  "roles": {}, "notifications": {"enabled": False},
                  "cleanup": {"interval": 1, "patterns": ["*.tmp"]}}
    runner.results_dir = results
    runner.lake = SimpleNamespace(reconcile_statuses=lambda *a, **kw: 0)
    runner.gpu_ids = []  # No executor is invoked in this maintenance-only test.
    runner.active, runner.active_evals = {}, {}
    runner.running, runner.iteration = True, 0
    runner.slot_mgr = SimpleNamespace(gpu_ids_in_use=lambda: set())
    runner._health_monitor = SimpleNamespace(check_before_write=lambda: True)
    for name in ("_startup_checks", "_kill_orphans", "_rebuild_config_hashes",
                 "_check_auto_upgrade", "_check_upgrade_sentinel",
                 "_check_cluster_versions"):
        setattr(runner, name, lambda: None)
    runner._check_stop_all = runner._check_disabled = lambda: False
    monkeypatch.setattr("orze.extensions.has_pro", lambda: False)
    monkeypatch.setattr(orchestrator, "notify", lambda *a: None)
    monkeypatch.setattr(orchestrator, "startup_canary", lambda *a: {})
    monkeypatch.setattr(orchestrator, "parse_ideas", lambda *a: {})
    monkeypatch.setattr(orchestrator, "_count_statuses", lambda *a, **kw: {})
    monkeypatch.setattr(orchestrator, "write_host_heartbeat", lambda *a: None)
    monkeypatch.setattr(orchestrator, "check_disk_space", lambda *a: True)
    monkeypatch.setattr(orchestrator, "_fs_lock", lambda *a, **kw: True)
    unlocked = []
    monkeypatch.setattr(orchestrator, "_fs_unlock", unlocked.append)
    observed = []

    def public_consumer(folder, cfg, *, lake=None):
        observed.append((folder, cfg, lake))
        raise MaintenanceReached

    monkeypatch.setattr(orchestrator, "run_cleanup", public_consumer)
    with pytest.raises(MaintenanceReached):
        runner._run_leased()

    assert observed == [(results, runner.cfg, runner.lake)]
    assert unlocked == [results / "_cleanup_lock"]
