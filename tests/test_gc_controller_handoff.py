"""New GC call-chain mechanisms: scope handoff, not destructive E2E proofs."""
from types import SimpleNamespace
from pathlib import Path

import pytest

from orze.agents import orze_gc
from orze.engine import orchestrator
from orze.idea_lake import IdeaLake


class GCReached(BaseException):
    """Stop after the real controller reaches the selected consumer boundary."""


def runner_for(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    lake = IdeaLake(tmp_path / "authority.db")
    runner = orchestrator.Orze.__new__(orchestrator.Orze)
    runner.cfg = {"_project_root": str(tmp_path), "ideas_file": str(tmp_path / "ideas.md"),
                  "results_dir": str(results), "idea_lake_db": str(lake.db_path),
                  "timeout": 60, "poll": 1, "roles": {},
                  "notifications": {"enabled": False},
                  "cleanup": {"interval": 1, "patterns": []},
                  "gc": {"enabled": True, "checkpoints_dir": str(checkpoints)},
                  "report": {"primary_metric": "quality", "sort": "ascending"},
                  "domain_contract": {"sentinel": ["full", "configuration"]}}
    runner.results_dir, runner.lake = results, lake
    runner.gpu_ids = []
    runner.active, runner.active_evals = {}, {}
    runner.running, runner.iteration = True, 0
    runner.slot_mgr = SimpleNamespace(gpu_ids_in_use=lambda: set(), free_gpu_ids=lambda **kw: [])
    runner._health_monitor = SimpleNamespace(check_before_write=lambda: True)
    return runner, checkpoints


def consumer_spy(monkeypatch):
    observed = []
    def consume(**kwargs):
        observed.append(kwargs)
        raise GCReached
    monkeypatch.setattr(orze_gc, "run_gc", consume)
    return observed


def assert_scope(observed, runner, checkpoints):
    assert len(observed) == 1
    assert observed[0]["cfg"] is runner.cfg
    assert observed[0]["lake"] is runner.lake
    assert observed[0]["results_dir"] == runner.results_dir
    assert observed[0]["checkpoints_dir"] == checkpoints
    assert observed[0]["lake_db_path"] == Path(runner.lake.db_path)
    assert observed[0]["cfg"]["domain_contract"] == {"sentinel": ["full", "configuration"]}


def test_real_periodic_loop_hands_full_scope_through_cleanup_to_gc(tmp_path, monkeypatch):
    runner, checkpoints = runner_for(tmp_path)
    for name in ("_startup_checks", "_kill_orphans", "_rebuild_config_hashes",
                 "_check_auto_upgrade", "_check_upgrade_sentinel", "_check_cluster_versions"):
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
    observed = consumer_spy(monkeypatch)
    try:
        with pytest.raises(GCReached):
            runner._run_leased()
        assert_scope(observed, runner, checkpoints)
        assert unlocked == [runner.results_dir / "_cleanup_lock"]
    finally:
        runner.lake.close()


def test_real_low_disk_launch_phase_hands_full_scope_and_active_ids_to_gc(tmp_path, monkeypatch):
    runner, checkpoints = runner_for(tmp_path)
    runner.active = {0: SimpleNamespace(idea_id="idea-training")}
    runner.active_evals = {1: SimpleNamespace(idea_id="idea-evaluation")}
    observed = consumer_spy(monkeypatch)
    try:
        with pytest.raises(GCReached):
            runner._launch_training([], False, {})
        assert_scope(observed, runner, checkpoints)
        assert observed[0]["extra_keep_ids"] == {"idea-training", "idea-evaluation"}
        assert observed[0]["min_free_gb"] == 0
    finally:
        runner.lake.close()
