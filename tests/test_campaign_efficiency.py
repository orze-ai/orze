import ast
import datetime
import inspect
import json
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

import orze.engine.orchestrator as orchestrator_module
import orze.engine.phases as phases_module
from orze.engine.campaign_efficiency import (
    DEFAULT_CAMPAIGN_TARGETS,
    DEFAULT_OUTCOME_TARGETS,
    analyze_campaign,
    capture_campaign_efficiency_sample,
    capture_campaign_progress_update,
    preregister_campaign,
    require_active_campaign_registration,
)
from orze.engine.reproducibility import config_identity_sha256
from orze.engine.orchestrator import Orze
from orze.engine.phases import OrzePhaseMixin
from orze.idea_lake import IdeaLake
from orze.core.config import _validate_config


def _iso(epoch):
    return datetime.datetime.fromtimestamp(
        epoch, datetime.timezone.utc
    ).isoformat()


def _telemetry(scope):
    return [
        {
            "index": gpu,
            "name": "test-gpu",
            "memory_used_mib": 100,
            "memory_total_mib": 1000,
            "utilization_pct": 95,
            "temperature_c": 60,
        }
        for gpu in scope
    ]


def _manifest():
    start = time.time() + 10
    return {
        "campaign_id": "campaign-test-001",
        "expected_idea_ids": ["idea-001", "idea-002"],
        "start_epoch": start,
        "end_epoch": start + 40,
        "physical_scope": [4, 5, 6, 7],
        "poll_seconds": 10.0,
        "minimum_samples": 5,
        "minimum_claims": 1,
        "minimum_release_to_claim_pairs": 1,
        "targets": dict(DEFAULT_CAMPAIGN_TARGETS),
    }


def _populate_complete_campaign(
        db_path, manifest, *, controllers=None, hosts=None):
    controllers = controllers or ["controller-a"] * 5
    hosts = hosts or ["host-a"] * 5
    assert len(controllers) == 5
    assert len(hosts) == 5
    lake = IdeaLake(str(db_path))
    for iteration, (controller, host) in enumerate(
            zip(controllers, hosts), start=1):
        at = manifest["start_epoch"] + (iteration - 1) * 10
        lake.record_harness_efficiency_sample(
            campaign_id=manifest["campaign_id"],
            controller_id=controller,
            host=host,
            iteration=iteration,
            observed_at_epoch=at,
            poll_seconds=10,
            physical_scope=[4, 5, 6, 7],
            gpu_telemetry=_telemetry([4, 5, 6, 7]),
            active_training_gpus=[4, 5, 6],
            active_evaluation_gpus=[7],
            remaining_training=2,
            remaining_evaluation=0,
            launcher_paused=False,
            disk_ok=True,
        )
        lake.record_harness_campaign_progress(
            campaign_id=manifest["campaign_id"],
            controller_id=controller,
            host=host,
            iteration=iteration,
            observed_at_epoch=at,
            last_valid_artifact_sha256=None,
            last_valid_artifact_idea_id=None,
            last_valid_artifact_at_epoch=None,
            blocker_code="training_active",
            next_deadline_epoch=min(
                manifest["end_epoch"],
                at + manifest["targets"]["max_operator_update_gap_seconds"],
            ),
        )
    lake.conn.executemany(
        "INSERT INTO idea_state "
        "(idea_id, current_state, queued_at, claimed_at) VALUES (?, ?, ?, ?)",
        [
            (
                "idea-001", "IN_PROGRESS", _iso(manifest["start_epoch"]),
                _iso(manifest["start_epoch"] + 5),
            ),
            (
                "idea-002", "CLAIMED", _iso(manifest["start_epoch"] + 15),
                _iso(manifest["start_epoch"] + 20),
            ),
        ],
    )
    lake.conn.executemany(
        "INSERT INTO idea_transitions "
        "(idea_id, from_state, to_state, ts) VALUES (?, ?, ?, ?)",
        [
            (
                "idea-001", "IN_PROGRESS", "COMPLETE",
                _iso(manifest["start_epoch"] + 15),
            ),
            (
                "idea-002", "QUEUED", "CLAIMED",
                _iso(manifest["start_epoch"] + 20),
            ),
        ],
    )
    lake.conn.commit()
    lake.close()


def test_capture_queries_only_explicit_physical_scope(tmp_path):
    lake = IdeaLake(str(tmp_path / "lake.db"))
    calls = []

    def query(scope):
        calls.append(scope)
        return _telemetry(scope)

    assert capture_campaign_efficiency_sample(
        lake,
        campaign_id=None,
        controller_id="controller-a",
        host="host-a",
        iteration=1,
        poll_seconds=10,
        physical_scope=[4, 5, 6, 7],
        active_training_gpus=[4, 5],
        active_evaluation_gpus=[6],
        remaining_training=1,
        remaining_evaluation=0,
        launcher_paused=False,
        disk_ok=True,
        observed_at_epoch=1000,
        telemetry_query=query,
    )
    assert calls == [[4, 5, 6, 7]]
    row = lake.conn.execute(
        "SELECT telemetry_complete, physical_scope_json "
        "FROM harness_efficiency_samples"
    ).fetchone()
    assert row["telemetry_complete"] == 1
    assert row["physical_scope_json"] == "[4,5,6,7]"
    lake.close()


def test_campaign_sample_duplicate_must_be_content_identical(tmp_path):
    lake = IdeaLake(str(tmp_path / "lake.db"))
    sample = {
        "campaign_id": "campaign-identity",
        "controller_id": "controller-a",
        "host": "host-a",
        "iteration": 1,
        "observed_at_epoch": 1000.0,
        "poll_seconds": 10.0,
        "physical_scope": [4, 5, 6, 7],
        "gpu_telemetry": _telemetry([4, 5, 6, 7]),
        "active_training_gpus": [4],
        "active_evaluation_gpus": [],
        "remaining_training": 1,
        "remaining_evaluation": 0,
        "launcher_paused": False,
        "disk_ok": True,
    }

    assert lake.record_harness_efficiency_sample(**sample) is True
    assert lake.record_harness_efficiency_sample(**sample) is True
    with pytest.raises(
            OSError, match="campaign_efficiency_database_identity_conflict"):
        lake.record_harness_efficiency_sample(
            **{**sample, "remaining_training": 2}
        )
    lake.close()


def test_required_campaign_sample_retains_but_rejects_incomplete_telemetry(
        tmp_path):
    lake = IdeaLake(str(tmp_path / "lake.db"))
    with pytest.raises(
            OSError, match="campaign_efficiency_telemetry_incomplete"):
        capture_campaign_efficiency_sample(
            lake,
            campaign_id="campaign-required",
            controller_id="controller-a",
            host="host-a",
            iteration=1,
            poll_seconds=10,
            physical_scope=[4, 5, 6, 7],
            active_training_gpus=[4],
            active_evaluation_gpus=[],
            remaining_training=1,
            remaining_evaluation=0,
            launcher_paused=False,
            disk_ok=True,
            observed_at_epoch=1000,
            require_complete_telemetry=True,
            telemetry_query=lambda scope: [],
        )
    row = lake.conn.execute(
        "SELECT telemetry_complete FROM harness_efficiency_samples"
    ).fetchone()
    assert row["telemetry_complete"] == 0
    lake.close()


def test_required_campaign_sample_failure_persistently_halts_controller(
        tmp_path, monkeypatch):
    runner = Orze.__new__(Orze)
    runner.cfg = {
        "poll": 10,
        "campaign_efficiency": {
            "enabled": True,
            "required_for_launch": True,
            "campaign_id": "campaign-required",
        },
        "launcher": {"paused": False},
    }
    runner.results_dir = tmp_path / "results"
    runner.lake = object()
    runner.gpu_ids = [4, 5, 6, 7]
    runner.slot_mgr = SimpleNamespace(gpu_ids_in_use=lambda: {4})
    runner.active_evals = {}
    runner.pending_evals = []
    runner._instance_uuid = "controller-a"
    runner._hostname = "host-a"
    runner.iteration = 3
    runner.running = True
    runner._stop_event = threading.Event()
    monkeypatch.setattr(
        orchestrator_module,
        "capture_campaign_efficiency_sample",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    assert runner._capture_campaign_efficiency_evidence([], [], True) is False

    pause = runner.results_dir / "_launcher_paused.flag"
    payload = json.loads(pause.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["reason"] == "campaign_efficiency_sample_failed"
    assert payload["campaign_id"] == "campaign-required"
    assert payload["iteration"] == 3
    assert runner.running is False
    assert runner._stop_kill_all is True
    assert runner._stop_event.is_set()


def test_required_campaign_sample_passes_complete_contract_to_sampler(
        tmp_path, monkeypatch):
    observed = {}
    runner = Orze.__new__(Orze)
    runner.cfg = {
        "poll": 10,
        "campaign_efficiency": {
            "enabled": True,
            "required_for_launch": True,
            "campaign_id": "campaign-required",
        },
        "launcher": {"paused": False},
    }
    runner.results_dir = tmp_path / "results"
    runner.lake = object()
    runner.gpu_ids = [4, 5, 6, 7]
    runner.slot_mgr = SimpleNamespace(gpu_ids_in_use=lambda: {4})
    runner.active_evals = {
        5: SimpleNamespace(idea_id="idea-eval-active")
    }
    runner.pending_evals = ["idea-eval-pending"]
    runner._instance_uuid = "controller-a"
    runner._hostname = "host-a"
    runner.iteration = 4

    def capture(*args, **kwargs):
        observed.update(kwargs)
        return True

    monkeypatch.setattr(
        orchestrator_module, "capture_campaign_efficiency_sample", capture
    )

    assert runner._capture_campaign_efficiency_evidence(
        ["idea-train-pending"],
        [(1, "idea-eval-active"), (2, "idea-eval-backlog")],
        True,
    ) is True
    assert observed["physical_scope"] == [4, 5, 6, 7]
    assert observed["active_training_gpus"] == [4]
    assert observed["active_evaluation_gpus"] == [5]
    assert observed["remaining_training"] == 1
    assert observed["remaining_evaluation"] == 2
    assert observed["require_complete_telemetry"] is True


def test_required_campaign_progress_failure_halts_dispatch(tmp_path, monkeypatch):
    halted = []
    runner = SimpleNamespace(
        cfg={
            "campaign_efficiency": {
                "enabled": True,
                "required_for_launch": True,
                "campaign_id": "campaign-required",
            },
            "launcher": {"paused": False},
            "report": {"primary_metric": "score"},
        },
        lake=object(),
        results_dir=tmp_path / "results",
        active_evals={},
        active={},
        pending_evals=[],
        _instance_uuid="controller-a",
        _hostname="host-a",
        iteration=2,
        _halt_required_campaign_evidence=halted.append,
    )
    monkeypatch.setattr(
        phases_module,
        "capture_campaign_progress_update",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    progress = OrzePhaseMixin._capture_campaign_progress_evidence(
        runner, [], [], True, []
    )

    assert progress["status"] == "UNAVAILABLE"
    assert halted == ["campaign_progress_update_failed"]


def test_managed_campaign_iterations_emit_matching_progress_evidence():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Orze._run_leased)))
    managed_branches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "managed_idea"
    ]
    assert any(
        any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "_capture_campaign_progress_evidence"
            for statement in branch.body
            for call in ast.walk(statement)
            if isinstance(call, ast.Call)
        )
        for branch in managed_branches
    )


def test_progress_update_is_visible_content_bound_and_persistent(tmp_path):
    db_path = tmp_path / "lake.db"
    results = tmp_path / "results"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    row = {
        "id": "idea-001",
        "primary_val": 0.75,
        "evidence_qualified": True,
        "evidence_reason": "benchmark_evidence_verified",
        "evidence_sha256": "e" * 64,
    }
    first = capture_campaign_progress_update(
        lake,
        results_dir=results,
        campaign_id=manifest["campaign_id"],
        controller_id="controller-a",
        host="host-a",
        iteration=1,
        completed_rows=[row],
        primary_metric="score",
        blocker_code="training_active",
        observed_at_epoch=manifest["start_epoch"],
    )
    second = capture_campaign_progress_update(
        lake,
        results_dir=results,
        campaign_id=manifest["campaign_id"],
        controller_id="controller-a",
        host="host-a",
        iteration=2,
        completed_rows=[row],
        primary_metric="score",
        blocker_code="evaluation_active",
        observed_at_epoch=manifest["start_epoch"] + 10,
    )

    assert first["last_valid_artifact_sha256"] == (
        second["last_valid_artifact_sha256"]
    )
    assert second["last_valid_artifact_at_epoch"] == manifest["start_epoch"]
    latest = json.loads((
        results / "_campaign_progress" / manifest["campaign_id"] / "latest.json"
    ).read_text(encoding="utf-8"))
    assert latest == second
    assert latest["blocker_code"] == "evaluation_active"
    assert latest["next_deadline_epoch"] <= manifest["end_epoch"]
    assert lake.conn.execute(
        "SELECT COUNT(*) FROM harness_campaign_progress"
    ).fetchone()[0] == 2
    lake.close()


def test_progress_update_identity_conflict_fails_closed(tmp_path):
    db_path = tmp_path / "lake.db"
    results = tmp_path / "results"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    kwargs = {
        "results_dir": results,
        "campaign_id": manifest["campaign_id"],
        "controller_id": "controller-a",
        "host": "host-a",
        "iteration": 1,
        "completed_rows": [],
        "primary_metric": "score",
        "observed_at_epoch": manifest["start_epoch"],
    }
    capture_campaign_progress_update(
        lake, blocker_code="training_active", **kwargs
    )
    with pytest.raises(OSError, match="identity_conflict"):
        capture_campaign_progress_update(
            lake, blocker_code="evaluation_active", **kwargs
        )
    lake.close()


def test_progress_update_rejects_redirected_output_directory(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    results = tmp_path / "results"
    results.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (results / "_campaign_progress").symlink_to(outside)

    with pytest.raises(OSError, match="directory_redirected"):
        capture_campaign_progress_update(
            lake,
            results_dir=results,
            campaign_id=manifest["campaign_id"],
            controller_id="controller-a",
            host="host-a",
            iteration=1,
            completed_rows=[],
            primary_metric="score",
            blocker_code="training_active",
            observed_at_epoch=manifest["start_epoch"],
        )
    assert lake.conn.execute(
        "SELECT COUNT(*) FROM harness_campaign_progress"
    ).fetchone()[0] == 0
    assert list(outside.iterdir()) == []
    lake.close()


def test_enabled_sampling_requires_campaign_id():
    errors, _ = _validate_config({
        "campaign_efficiency": {"enabled": True},
    })
    assert "campaign_efficiency.campaign_id: required when enabled" in errors


def test_required_campaign_evidence_blocks_unmeasured_unpause():
    errors, _ = _validate_config({
        "launcher": {"paused": False},
        "campaign_efficiency": {
            "enabled": False,
            "campaign_id": None,
            "required_for_launch": True,
        },
    })
    assert any("required before unpausing" in error for error in errors)

    paused_errors, _ = _validate_config({
        "launcher": {"paused": True},
        "campaign_efficiency": {
            "enabled": False,
            "campaign_id": None,
            "required_for_launch": True,
        },
    })
    assert not any("required before unpausing" in error
                   for error in paused_errors)


def test_active_campaign_registration_is_required_before_launch(tmp_path):
    db_path = tmp_path / "lake.db"
    lake = IdeaLake(str(db_path))
    with pytest.raises(RuntimeError, match="registration_missing"):
        require_active_campaign_registration(
            lake,
            campaign_id="campaign-test-001",
            physical_scope=[4, 5, 6, 7],
            poll_seconds=10,
            now_epoch=time.time(),
        )
    lake.close()

    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    report = require_active_campaign_registration(
        lake,
        campaign_id=manifest["campaign_id"],
        physical_scope=[4, 5, 6, 7],
        poll_seconds=10,
        now_epoch=manifest["start_epoch"],
    )
    assert report["manifest_sha256"]
    assert report["expected_idea_count"] == 2
    with pytest.raises(RuntimeError, match="physical_scope_mismatch"):
        require_active_campaign_registration(
            lake,
            campaign_id=manifest["campaign_id"],
            physical_scope=[4, 5, 6],
            poll_seconds=10,
            now_epoch=manifest["start_epoch"],
        )
    with pytest.raises(RuntimeError, match="poll_seconds_mismatch"):
        require_active_campaign_registration(
            lake,
            campaign_id=manifest["campaign_id"],
            physical_scope=[4, 5, 6, 7],
            poll_seconds=11,
            now_epoch=manifest["start_epoch"],
        )
    with pytest.raises(RuntimeError, match="window_inactive"):
        require_active_campaign_registration(
            lake,
            campaign_id=manifest["campaign_id"],
            physical_scope=[4, 5, 6, 7],
            poll_seconds=10,
            now_epoch=manifest["end_epoch"] + 1,
        )
    lake.close()


def test_campaign_id_must_be_safe_for_operator_receipt_path():
    errors, _ = _validate_config({
        "campaign_efficiency": {
            "enabled": True,
            "campaign_id": "../redirect",
        },
    })
    assert any("safe non-empty identifier" in error for error in errors)


def test_registration_rejects_weakened_targets(tmp_path):
    manifest = _manifest()
    manifest["targets"]["min_allocation_duty_cycle"] = 0.89
    with pytest.raises(ValueError, match="cannot be weaker"):
        preregister_campaign(tmp_path / "lake.db", manifest)


def test_registration_rejects_weakened_outcome_targets(tmp_path):
    manifest = _manifest()
    manifest["outcome_contract"] = {
        "expected_decision_identity_sha256": ["a" * 64],
        "artifact_relation": "identical",
        "reproducibility_contract": {
            "mode": "not_applicable",
            "rationale": (
                "This synthetic campaign does not ask a replication question."
            ),
            "expected_config_identity_sha256": {
                idea_id: config_identity_sha256({"seed": index})
                for index, idea_id in enumerate(
                    manifest["expected_idea_ids"], start=1
                )
            },
        },
        "targets": dict(DEFAULT_OUTCOME_TARGETS),
    }
    manifest["outcome_contract"]["targets"][
        "max_gpu_hours_per_qualified_success"
    ] = 8.1
    with pytest.raises(ValueError, match="cannot be weaker"):
        preregister_campaign(tmp_path / "lake.db", manifest)


def test_incomplete_telemetry_is_retained_and_fails_closed(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    for iteration in range(1, 6):
        at = manifest["start_epoch"] + (iteration - 1) * 10
        lake.record_harness_efficiency_sample(
            campaign_id=manifest["campaign_id"],
            controller_id="controller-a",
            host="host-a",
            iteration=iteration,
            observed_at_epoch=at,
            poll_seconds=10,
            physical_scope=[4, 5, 6, 7],
            gpu_telemetry=_telemetry([4, 5]),
            active_training_gpus=[4],
            active_evaluation_gpus=[],
            remaining_training=1,
            remaining_evaluation=0,
            launcher_paused=False,
            disk_ok=True,
        )
        lake.record_harness_campaign_progress(
            campaign_id=manifest["campaign_id"],
            controller_id="controller-a",
            host="host-a",
            iteration=iteration,
            observed_at_epoch=at,
            last_valid_artifact_sha256=None,
            last_valid_artifact_idea_id=None,
            last_valid_artifact_at_epoch=None,
            blocker_code="training_active",
            next_deadline_epoch=min(
                manifest["end_epoch"],
                at + manifest["targets"]["max_operator_update_gap_seconds"],
            ),
        )
    lake.close()

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )
    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["telemetry_complete"]["passed"] is False


def test_preregistered_complete_campaign_can_be_verified(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    for iteration in range(1, 6):
        at = manifest["start_epoch"] + (iteration - 1) * 10
        lake.record_harness_efficiency_sample(
            campaign_id=manifest["campaign_id"],
            controller_id="controller-a",
            host="host-a",
            iteration=iteration,
            observed_at_epoch=at,
            poll_seconds=10,
            physical_scope=[4, 5, 6, 7],
            gpu_telemetry=_telemetry([4, 5, 6, 7]),
            active_training_gpus=[4, 5, 6],
            active_evaluation_gpus=[7],
            remaining_training=2,
            remaining_evaluation=0,
            launcher_paused=False,
            disk_ok=True,
        )
        lake.record_harness_campaign_progress(
            campaign_id=manifest["campaign_id"],
            controller_id="controller-a",
            host="host-a",
            iteration=iteration,
            observed_at_epoch=at,
            last_valid_artifact_sha256=None,
            last_valid_artifact_idea_id=None,
            last_valid_artifact_at_epoch=None,
            blocker_code="training_active",
            next_deadline_epoch=min(
                manifest["end_epoch"],
                at + manifest["targets"]["max_operator_update_gap_seconds"],
            ),
        )
    lake.conn.execute(
        "INSERT INTO idea_state "
        "(idea_id, current_state, queued_at, claimed_at) VALUES (?, ?, ?, ?)",
        (
            "idea-001", "IN_PROGRESS", _iso(manifest["start_epoch"]),
            _iso(manifest["start_epoch"] + 5),
        ),
    )
    lake.conn.execute(
        "INSERT INTO idea_state "
        "(idea_id, current_state, queued_at, claimed_at) VALUES (?, ?, ?, ?)",
        (
            "idea-002", "CLAIMED", _iso(manifest["start_epoch"] + 15),
            _iso(manifest["start_epoch"] + 20),
        ),
    )
    lake.conn.executemany(
        "INSERT INTO idea_transitions "
        "(idea_id, from_state, to_state, ts) VALUES (?, ?, ?, ?)",
        [
            (
                "idea-001", "IN_PROGRESS", "COMPLETE",
                _iso(manifest["start_epoch"] + 15),
            ),
            (
                "idea-002", "QUEUED", "CLAIMED",
                _iso(manifest["start_epoch"] + 20),
            ),
        ],
    )
    lake.conn.commit()
    lake.close()

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )
    assert receipt["status"] == "VERIFIED"
    assert receipt["metrics"]["sample_count"] == 5
    assert receipt["metrics"]["allocation_duty_cycle"] == 1.0
    assert receipt["metrics"]["allocated_gpu_utilization_mean_pct"] == 95.0
    assert receipt["metrics"]["queue_to_claim_p95_seconds"] == 5.0
    assert receipt["metrics"]["terminal_to_next_claim_p95_seconds"] == 5.0


def test_one_claim_cannot_fill_multiple_terminal_latency_samples(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    _populate_complete_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    lake.conn.execute(
        "INSERT INTO idea_transitions "
        "(idea_id, from_state, to_state, ts) VALUES (?, ?, ?, ?)",
        (
            "idea-002", "IN_PROGRESS", "FAILED",
            _iso(manifest["start_epoch"] + 16),
        ),
    )
    lake.conn.commit()
    lake.close()

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )

    assert receipt["metrics"]["terminal_release_count"] == 2
    assert receipt["metrics"]["terminal_to_next_claim_count"] == 1
    assert receipt["metrics"]["unmatched_terminal_release_count"] == 1


def test_retry_cannot_hide_an_earlier_slow_queue_to_claim(tmp_path):
    """Every immutable claim must retain its own eligible-queue latency."""
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    _populate_complete_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    lake.conn.execute("DELETE FROM idea_transitions")
    lake.conn.execute("DELETE FROM idea_state")
    for idea_id in manifest["expected_idea_ids"]:
        lake.insert(
            idea_id, "test", "{}", "", status="queued",
            created_at=_iso(manifest["start_epoch"]),
        )
    lake.conn.execute(
        "UPDATE idea_state SET current_state = 'CLAIMED', queued_at = ?, "
        "claimed_at = ? WHERE idea_id = 'idea-001'",
        (
            _iso(manifest["start_epoch"] + 32),
            _iso(manifest["start_epoch"] + 33),
        ),
    )
    lake.conn.execute(
        "UPDATE idea_state SET current_state = 'CLAIMED', queued_at = ?, "
        "claimed_at = ? WHERE idea_id = 'idea-002'",
        (
            _iso(manifest["start_epoch"] + 15),
            _iso(manifest["start_epoch"] + 20),
        ),
    )
    lake.conn.executemany(
        "INSERT INTO idea_transitions "
        "(idea_id, from_state, to_state, ts) VALUES (?, ?, ?, ?)",
        [
            ("idea-002", "QUEUED", "CLAIMED",
             _iso(manifest["start_epoch"] + 20)),
            ("idea-001", "QUEUED", "CLAIMED",
             _iso(manifest["start_epoch"] + 30)),
            ("idea-001", "CLAIMED", "IN_PROGRESS",
             _iso(manifest["start_epoch"] + 30.5)),
            ("idea-001", "IN_PROGRESS", "FAILED",
             _iso(manifest["start_epoch"] + 31)),
            ("idea-001", "FAILED", "QUEUED",
             _iso(manifest["start_epoch"] + 32)),
            ("idea-001", "QUEUED", "CLAIMED",
             _iso(manifest["start_epoch"] + 33)),
        ],
    )
    lake.conn.commit()
    lake.close()

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )

    assert receipt["metrics"]["queue_to_claim_count"] == 3
    assert receipt["metrics"]["queue_to_claim_p95_seconds"] > 20.0
    assert receipt["checks"]["queue_to_claim"]["passed"] is False

    lake = IdeaLake(str(db_path))
    lake.conn.execute(
        "UPDATE idea_state SET first_queued_at = NULL "
        "WHERE idea_id = 'idea-001'"
    )
    lake.conn.commit()
    lake.close()
    incomplete = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )
    assert incomplete["status"] == "UNVERIFIED"
    assert incomplete["metrics"]["unmatched_queue_to_claim_count"] == 1
    assert incomplete["checks"]["queue_claim_history_complete"][
        "passed"
    ] is False


def test_unrelated_lifecycle_rows_cannot_improve_campaign_latency(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    _populate_complete_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    lake.conn.execute(
        "INSERT INTO idea_state "
        "(idea_id, current_state, queued_at, claimed_at) VALUES (?, ?, ?, ?)",
        (
            "idea-unrelated", "CLAIMED", _iso(manifest["start_epoch"] + 24),
            _iso(manifest["start_epoch"] + 25),
        ),
    )
    lake.conn.commit()
    lake.close()

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["exact_campaign_idea_universe"]["passed"] is False
    assert receipt["metrics"]["unexpected_lifecycle_idea_ids"] == [
        "idea-unrelated"
    ]
    assert receipt["metrics"]["queue_to_claim_count"] == 2


def test_missing_preregistered_lifecycle_row_fails_closed(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    manifest["expected_idea_ids"].append("idea-003")
    preregister_campaign(db_path, manifest)
    _populate_complete_campaign(db_path, manifest)

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["exact_campaign_idea_universe"]["passed"] is False
    assert receipt["metrics"]["missing_lifecycle_idea_ids"] == ["idea-003"]


def test_multiple_sequential_controller_identities_are_observed(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    _populate_complete_campaign(
        db_path,
        manifest,
        controllers=[
            "controller-a", "controller-b", "controller-a",
            "controller-b", "controller-a",
        ],
    )

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )

    assert receipt["status"] == "VERIFIED"
    assert receipt["metrics"]["controller_count"] == 2
    assert receipt["checks"]["single_physical_host"]["passed"] is True


def test_multiple_physical_hosts_fail_closed(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    _populate_complete_campaign(
        db_path,
        manifest,
        controllers=["controller-a"] * 5,
        hosts=["host-a", "host-b", "host-a", "host-b", "host-a"],
    )

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["metrics"]["host_count"] == 2
    assert receipt["checks"]["single_physical_host"]["passed"] is False


def test_missing_operator_update_fails_closed(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    _populate_complete_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    lake.conn.execute(
        "DELETE FROM harness_campaign_progress WHERE iteration = 3"
    )
    lake.conn.commit()
    lake.close()

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["operator_updates_complete"]["passed"] is False
    assert receipt["metrics"]["operator_update_count"] == 4


def test_operator_update_sla_miss_is_a_failed_target(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    manifest["targets"]["max_operator_update_gap_seconds"] = 5.0
    preregister_campaign(db_path, manifest)
    _populate_complete_campaign(db_path, manifest)

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )

    assert receipt["status"] == "FAILED"
    assert receipt["checks"]["operator_updates_complete"]["passed"] is True
    assert receipt["checks"]["operator_update_gap"]["passed"] is False
    assert receipt["metrics"]["max_operator_update_gap_seconds"] == 10.0


def test_operator_deadline_tamper_fails_closed(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    _populate_complete_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    lake.conn.execute(
        "UPDATE harness_campaign_progress SET next_deadline_epoch = ? "
        "WHERE iteration = 2",
        (manifest["end_epoch"] + 1,),
    )
    lake.conn.commit()
    lake.close()

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["operator_update_deadlines_valid"][
        "passed"
    ] is False


def test_operator_update_must_match_scheduler_sample_identity(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    _populate_complete_campaign(db_path, manifest)
    shifted = manifest["start_epoch"] + 11
    lake = IdeaLake(str(db_path))
    lake.conn.execute(
        "UPDATE harness_campaign_progress SET observed_at_epoch = ?, "
        "observed_at = ?, next_deadline_epoch = ? WHERE iteration = 2",
        (
            shifted,
            _iso(shifted),
            min(
                manifest["end_epoch"],
                shifted
                + manifest["targets"]["max_operator_update_gap_seconds"],
            ),
        ),
    )
    lake.conn.commit()
    lake.close()

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["operator_updates_complete"]["passed"] is False


def test_registration_rejects_invalid_campaign_idea_universe(tmp_path):
    manifest = _manifest()
    manifest["expected_idea_ids"] = ["idea-001", "idea-001"]
    with pytest.raises(ValueError, match="unique valid idea IDs"):
        preregister_campaign(tmp_path / "lake.db", manifest)


def test_complete_evidence_with_missed_target_is_failed(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    preregister_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    for iteration in range(1, 6):
        at = manifest["start_epoch"] + (iteration - 1) * 10
        lake.record_harness_efficiency_sample(
            campaign_id=manifest["campaign_id"],
            controller_id="controller-a",
            host="host-a",
            iteration=iteration,
            observed_at_epoch=at,
            poll_seconds=10,
            physical_scope=[4, 5, 6, 7],
            gpu_telemetry=_telemetry([4, 5, 6, 7]),
            active_training_gpus=[4],
            active_evaluation_gpus=[],
            remaining_training=4,
            remaining_evaluation=0,
            launcher_paused=False,
            disk_ok=True,
        )
        lake.record_harness_campaign_progress(
            campaign_id=manifest["campaign_id"],
            controller_id="controller-a",
            host="host-a",
            iteration=iteration,
            observed_at_epoch=at,
            last_valid_artifact_sha256=None,
            last_valid_artifact_idea_id=None,
            last_valid_artifact_at_epoch=None,
            blocker_code="eligible_queue_waiting",
            next_deadline_epoch=min(
                manifest["end_epoch"],
                at + manifest["targets"]["max_operator_update_gap_seconds"],
            ),
        )
    lake.conn.execute(
        "INSERT INTO idea_state "
        "(idea_id, current_state, queued_at, claimed_at) VALUES (?, ?, ?, ?)",
        (
            "idea-001", "IN_PROGRESS", _iso(manifest["start_epoch"]),
            _iso(manifest["start_epoch"] + 5),
        ),
    )
    lake.conn.execute(
        "INSERT INTO idea_state "
        "(idea_id, current_state, queued_at, claimed_at) VALUES (?, ?, ?, ?)",
        (
            "idea-002", "CLAIMED", _iso(manifest["start_epoch"] + 15),
            _iso(manifest["start_epoch"] + 20),
        ),
    )
    lake.conn.executemany(
        "INSERT INTO idea_transitions "
        "(idea_id, from_state, to_state, ts) VALUES (?, ?, ?, ?)",
        [
            (
                "idea-001", "IN_PROGRESS", "COMPLETE",
                _iso(manifest["start_epoch"] + 15),
            ),
            (
                "idea-002", "QUEUED", "CLAIMED",
                _iso(manifest["start_epoch"] + 20),
            ),
        ],
    )
    lake.conn.commit()
    lake.close()

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )
    assert receipt["status"] == "FAILED"
    assert receipt["checks"]["allocation_duty_cycle"]["passed"] is False


def test_tampered_sample_is_unverified_instead_of_crashing(tmp_path):
    db_path = tmp_path / "lake.db"
    manifest = _manifest()
    manifest["minimum_samples"] = 1
    preregister_campaign(db_path, manifest)
    lake = IdeaLake(str(db_path))
    lake.record_harness_efficiency_sample(
        campaign_id=manifest["campaign_id"],
        controller_id="controller-a",
        host="host-a",
        iteration=1,
        observed_at_epoch=manifest["start_epoch"],
        poll_seconds=10,
        physical_scope=[4, 5, 6, 7],
        gpu_telemetry=_telemetry([4, 5, 6, 7]),
        active_training_gpus=[4],
        active_evaluation_gpus=[],
        remaining_training=1,
        remaining_evaluation=0,
        launcher_paused=False,
        disk_ok=True,
    )
    lake.conn.execute(
        "UPDATE harness_efficiency_samples SET gpu_telemetry_json = ?",
        (json.dumps([{"index": "not-an-integer"}]),),
    )
    lake.conn.commit()
    lake.close()

    receipt = analyze_campaign(
        db_path, manifest, now_epoch=manifest["end_epoch"] + 10
    )
    assert receipt["status"] == "UNVERIFIED"
    assert receipt["metrics"]["malformed_sample_count"] == 1
    assert receipt["checks"]["no_malformed_samples"]["passed"] is False
