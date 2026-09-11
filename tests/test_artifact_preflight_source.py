"""Native receipt/report mechanisms, not preceding-version behavior failures.

The resolver is a real tiny CPU process. The started-consumer case constructs
explicit SQLite protocol metadata to isolate this validator; root's separate
public launch tests cover actual READY/GO. No real resolver/training runs.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest

from orze.core.execution_attempts import create_attempt, current_attempt, mark_running
from orze.engine import artifact_preflight_receipts as receipts, scheduler
from orze.engine import artifact_preflight_failure_report as reports
from orze.engine.execution_authority import lifecycle_fence
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_artifact_preflight_tree_completion import cpu_preflight
from test_native_pre_script_tree_completion import cpu_pre_script


@pytest.fixture
def resolver(cpu_preflight, tmp_path):
    c = cpu_preflight
    script = tmp_path / "tiny_resolver.py"
    script.write_text("raise SystemExit(0)\n")
    c.cfg["artifact_preflight"].update(script=str(script), args=[])
    assert scheduler.claim(c.idea, c.results, 4, lake=c.lake)
    c.resolver = script
    c.claim = json.loads((c.folder / "claim.json").read_text())
    return c


def _run(c):
    from orze.engine.native_artifact_preflight import run_native_artifact_preflight
    return run_native_artifact_preflight(c.idea, c.results, c.cfg, c.lake)


def _passed(c):
    result = _run(c)
    assert result
    capture = receipts.capture_preflight_source(c.lake, c.folder, c.cfg)
    assert capture.source["attempt_ref"] == asdict(result.attempt_ref)
    return result, capture


def test_passed_capture_is_detached_and_writer_verification_does_not_hash_files(resolver, monkeypatch):
    c = resolver
    result, capture = _passed(c)
    source = capture.source
    assert source["claim_attempt_id"] == c.claim["attempt_id"] != result.attempt_ref.attempt_id
    source["claim_sha256"] = "0" * 64
    assert capture.source != source
    def forbidden(*args, **kwargs):
        pytest.fail("large input hashing entered the SQLite writer")
    monkeypatch.setattr(receipts, "_identity", forbidden)
    c.lake.conn.execute("BEGIN IMMEDIATE")
    try:
        assert receipts.verify_preflight_source(c.lake, c.folder, c.cfg, capture,
            check_inputs=False) == capture.source
    finally:
        c.lake.conn.rollback()
    with pytest.raises(TerminationUnconfirmed):
        receipts.verify_preflight_source(c.lake, c.folder, c.cfg, capture, check_inputs=False)


def test_go_rechecks_input_bytes_and_rejects_changed_resolver(resolver):
    c = resolver
    _, capture = _passed(c)
    c.resolver.write_text("raise SystemExit(7)\n")
    with pytest.raises(TerminationUnconfirmed):
        receipts.verify_preflight_source(c.lake, c.folder, c.cfg, capture)
    assert not (c.folder / "_compute_receipts").exists()


def test_copied_static_receipt_without_native_source_cannot_authorize(resolver):
    c = resolver
    _passed(c)
    other = "idea-static-copy"
    c.lake.insert(other, "Synthetic second task", "seed: 13\n", "", status="queued")
    assert scheduler.claim(other, c.results, 4, lake=c.lake)
    target = c.results / other
    value = json.loads((c.folder / "artifact_preflight.json").read_text())
    value["idea_id"] = other
    (target / "artifact_preflight.json").write_text(json.dumps(value))
    with pytest.raises(TerminationUnconfirmed):
        receipts.capture_preflight_source(c.lake, target, c.cfg)
    assert current_attempt(c.lake.conn, other, "artifact_preflight") is None
    assert not (target / "_compute_receipts").exists()


def test_new_claim_cannot_reuse_previous_claim_receipt(resolver):
    c = resolver
    _, capture = _passed(c)
    from orze.engine.failure import _reset_idea_for_retry
    _reset_idea_for_retry(c.folder, release_claim=True, lake=c.lake)
    assert c.lake.record_state_transition(c.idea, "CLAIMED", "QUEUED")
    assert scheduler.claim(c.idea, c.results, 4, lake=c.lake)
    assert json.loads((c.folder / "claim.json").read_text())["attempt_id"] != c.claim["attempt_id"]
    with pytest.raises(TerminationUnconfirmed):
        receipts.capture_preflight_source(c.lake, c.folder, c.cfg)
    with pytest.raises(TerminationUnconfirmed):
        receipts.verify_preflight_source(c.lake, c.folder, c.cfg, capture)


@pytest.mark.parametrize("phase", ["training", "posthoc"])
def test_started_consumer_uses_original_binding_not_obsolete_claim_bytes(resolver, phase):
    c = resolver
    _, capture = _passed(c)
    source = capture.source
    c.lake.conn.execute("BEGIN IMMEDIATE")
    consumer = create_attempt(c.lake.conn, c.idea, phase, c.claim["attempt_id"], {})
    assert c.lake._record_state_transition_in_tx(c.idea, "CLAIMED", "IN_PROGRESS",
        reason="explicit validator metadata fixture", sop_type="training")
    binding = {"origin": "native_" + phase, "claim_sha256": source["claim_sha256"],
               "artifact_preflight_source": source,
               "supervision": {"identity": {"attempt_ref": asdict(consumer), "scope": str(c.folder)}},
               "lifecycle": lifecycle_fence(c.lake, c.idea, "training")}
    mark_running(c.lake.conn, consumer, binding)
    c.lake.conn.commit()
    claim = dict(c.claim, trainer_pid=1234, trainer_start_ticks=4567,
                 trainer_pgid=1234, trainer_started_at=1.0)
    (c.folder / "claim.json").write_text(json.dumps(claim))
    assert receipts.verify_preflight_source(c.lake, c.folder, c.cfg, capture,
        consumer_ref=consumer) == source
    with pytest.raises(TerminationUnconfirmed):
        receipts.verify_preflight_source(c.lake, c.folder, c.cfg, capture)
    corrupted = dict(binding, artifact_preflight_source=dict(source, receipt_sha256="0" * 64))
    c.lake.conn.execute("UPDATE execution_attempts SET binding_json=? WHERE attempt_id=?",
        (json.dumps(corrupted, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
         consumer.attempt_id))
    c.lake.conn.commit()
    with pytest.raises(TerminationUnconfirmed):
        receipts.verify_preflight_source(c.lake, c.folder, c.cfg, capture, consumer_ref=consumer)


@pytest.mark.parametrize("configured_failure", [False, True])
def test_failed_or_known_not_started_source_reports_once(resolver, monkeypatch, configured_failure):
    c = resolver
    if configured_failure:
        c.cfg["artifact_preflight"]["network"] = "required"
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:
        c.resolver.write_text("raise SystemExit(7)\n")
    result = _run(c)
    assert not result
    source = current_attempt(c.lake.conn, c.idea, "artifact_preflight")
    assert source["state"] == ("NOT_STARTED" if configured_failure else "TERMINAL")
    if configured_failure:
        assert c.roots == []
        assert source["terminal"]["process_tree"] is None
        assert source["terminal"]["return_code"] is None
    counts = {}
    answer = reports.report_artifact_preflight_failure(c.lake, c.folder,
        result.attempt_ref, counts, c.cfg)
    assert answer["status"] == "reported" and counts == {c.idea: 1}
    action = current_attempt(c.lake.conn, c.idea, reports.PHASE)
    assert action["attempt_id"].startswith("artifact-preflight-failure-")
    assert action["terminal"]["repair_status"] == "pending_explicit_action"
    assert c.lake.get_fsm_state(c.idea) == "FAILED"
    receipt = json.loads((c.folder / "_compute_receipts" / c.claim["attempt_id"] / "terminal.json").read_text())
    assert receipt["reason_code"] == "artifact_preflight_failed"
    assert receipt["allocated_gpu_seconds"] == 0.0
    counts.clear()  # Same-process projection test, not controller adoption.
    assert reports.report_artifact_preflight_failure(c.lake, c.folder,
        result.attempt_ref, counts, c.cfg)["status"] == "duplicate"
    assert counts == {c.idea: 1}
    assert c.events == []


def test_disabled_no_history_does_not_require_a_new_resolver(resolver):
    c = resolver
    c.cfg["artifact_preflight"]["enabled"] = False
    capture = receipts.capture_preflight_source(c.lake, c.folder, c.cfg)
    assert capture.source is None
    assert receipts.verify_preflight_source(c.lake, c.folder, c.cfg, capture) is None
    assert c.roots == []
    assert current_attempt(c.lake.conn, c.idea, "artifact_preflight") is None
