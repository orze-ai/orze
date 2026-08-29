import datetime
import json
import time

import pytest

from orze.engine.campaign_efficiency import (
    DEFAULT_CAMPAIGN_TARGETS,
    DEFAULT_OUTCOME_TARGETS,
    analyze_campaign,
    capture_campaign_efficiency_sample,
    preregister_campaign,
)
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
        "start_epoch": start,
        "end_epoch": start + 40,
        "physical_scope": [4, 5, 6, 7],
        "poll_seconds": 10.0,
        "minimum_samples": 5,
        "minimum_claims": 1,
        "minimum_release_to_claim_pairs": 1,
        "targets": dict(DEFAULT_CAMPAIGN_TARGETS),
    }


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


def test_enabled_sampling_requires_campaign_id():
    errors, _ = _validate_config({
        "campaign_efficiency": {"enabled": True},
    })
    assert "campaign_efficiency.campaign_id: required when enabled" in errors


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
    lake.conn.execute(
        "INSERT INTO idea_state "
        "(idea_id, current_state, queued_at, claimed_at) VALUES (?, ?, ?, ?)",
        (
            "idea-001", "IN_PROGRESS", _iso(manifest["start_epoch"]),
            _iso(manifest["start_epoch"] + 5),
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
    lake.conn.execute(
        "INSERT INTO idea_state "
        "(idea_id, current_state, queued_at, claimed_at) VALUES (?, ?, ?, ?)",
        (
            "idea-001", "IN_PROGRESS", _iso(manifest["start_epoch"]),
            _iso(manifest["start_epoch"] + 5),
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
