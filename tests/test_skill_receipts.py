"""Tests for orze.skills.receipts."""
import time

from orze.skills.receipts import (
    Receipt,
    compute_evidenced_skills,
    read_receipt,
    snapshot_mtimes,
    write_receipt,
)


def test_receipt_roundtrip(tmp_path):
    receipt = Receipt(
        role="data_analyst",
        cycle=42,
        started_at=1000.0,
        ended_at=1060.0,
        skills_declared=["sop-da-base", "sop-da-anomaly"],
        skills_evidenced=["sop-da-base"],
        outputs={"sop-da-base": ["results/_data_audit.md"]},
    )
    path = tmp_path / "_receipts" / "data_analyst_cycle42.json"
    write_receipt(receipt, path)
    loaded = read_receipt(path)
    assert loaded.role == "data_analyst"
    assert loaded.cycle == 42
    assert loaded.skills_declared == ["sop-da-base", "sop-da-anomaly"]
    assert loaded.skills_evidenced == ["sop-da-base"]
    assert "sop-da-anomaly" not in loaded.skills_evidenced


def test_evidence_detects_file_mtime_delta(tmp_path):
    declared = {
        "skill_a": ["out_a.md"],
        "skill_b": ["out_b.md"],
    }
    (tmp_path / "out_a.md").write_text("initial")
    (tmp_path / "out_b.md").write_text("initial")
    snap = snapshot_mtimes(declared, tmp_path)
    # simulate role writing to out_a but NOT out_b
    time.sleep(0.01)
    (tmp_path / "out_a.md").write_text("updated")
    evidenced = compute_evidenced_skills(declared, snap, tmp_path)
    assert "skill_a" in evidenced
    assert "skill_b" not in evidenced


def test_evidence_handles_missing_output_as_no_evidence(tmp_path):
    declared = {"skill_x": ["never_created.md"]}
    snap = snapshot_mtimes(declared, tmp_path)
    # no file writes happen
    evidenced = compute_evidenced_skills(declared, snap, tmp_path)
    assert evidenced == []


def test_evidence_detects_newly_created_file(tmp_path):
    declared = {"skill_fresh": ["brand_new.md"]}
    snap = snapshot_mtimes(declared, tmp_path)
    # File didn't exist at snapshot — mtime 0.0
    assert snap[str(tmp_path / "brand_new.md")] == 0.0
    (tmp_path / "brand_new.md").write_text("fresh content")
    evidenced = compute_evidenced_skills(declared, snap, tmp_path)
    assert "skill_fresh" in evidenced


def test_evidence_any_path_change_counts(tmp_path):
    """A skill producing multiple files is evidenced if ANY changes."""
    declared = {"skill_multi": ["a.md", "b.md"]}
    (tmp_path / "a.md").write_text("initial")
    (tmp_path / "b.md").write_text("initial")
    snap = snapshot_mtimes(declared, tmp_path)
    time.sleep(0.01)
    # only change one of the two
    (tmp_path / "a.md").write_text("changed")
    evidenced = compute_evidenced_skills(declared, snap, tmp_path)
    assert "skill_multi" in evidenced
