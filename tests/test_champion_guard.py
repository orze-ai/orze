"""Tests for orze.engine.champion_guard (F14)."""

import json
from pathlib import Path

from orze.engine.champion_guard import (
    GuardConfig,
    check_promotion,
    create_audit_idea,
    load_history,
    save_history,
)
from orze.idea_lake import IdeaLake


def _write_metrics(results_dir: Path, idea_id: str, metrics: dict):
    d = results_dir / idea_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics))


def test_guardconfig_reads_defaults():
    g = GuardConfig.from_cfg({})
    assert g.enabled is True
    assert g.z_threshold == 4.0
    assert g.min_history == 10


def test_guardconfig_overrides():
    g = GuardConfig.from_cfg({"champion_guard": {
        "z_threshold": 3.0, "min_history": 5, "enabled": False,
    }})
    assert g.z_threshold == 3.0
    assert g.min_history == 5
    assert g.enabled is False


def test_allows_promotion_without_history(tmp_path):
    _write_metrics(tmp_path, "idea-a", {"pgmAP_ALL": 0.9})
    ok, info = check_promotion(tmp_path, "idea-a", 0.9, {})
    assert ok is True
    assert info["blocked"] is False


def test_allows_plausible_improvement(tmp_path):
    # Pretend we've been promoting ideas in the 0.88-0.90 range for a while.
    save_history(tmp_path, [0.885, 0.887, 0.89, 0.891, 0.89,
                            0.893, 0.894, 0.895, 0.896, 0.898])
    _write_metrics(tmp_path, "idea-b", {"pgmAP_ALL": 0.9057})
    ok, info = check_promotion(tmp_path, "idea-b", 0.9057, {})
    assert ok is True
    assert info["blocked"] is False
    assert info["z"] is not None
    assert info["z"] < 4.0


def test_blocks_5_sigma_jump(tmp_path):
    # Calm distribution with small std → a huge jump is clearly 5σ.
    save_history(tmp_path, [0.88, 0.89, 0.881, 0.884, 0.883,
                            0.886, 0.887, 0.888, 0.889, 0.89])
    _write_metrics(tmp_path, "idea-c", {"pgmAP_ALL": 0.999})
    audit_created = []
    notified = []
    ok, info = check_promotion(
        tmp_path, "idea-c", 0.999, {},
        notify_fn=lambda kind, payload, cfg: notified.append((kind, payload)),
        create_audit_idea_fn=lambda aid, sid, p: audit_created.append(aid),
    )
    assert ok is False
    assert info["blocked"] is True
    assert info["z"] > 4.0
    assert info["audit_idea_id"] is not None
    assert notified and notified[0][0] == "audit"
    assert audit_created and audit_created[0].startswith("audit-idea-c")


def test_disabling_guard_allows_anything(tmp_path):
    save_history(tmp_path, [0.88] * 20)
    _write_metrics(tmp_path, "idea-d", {"pgmAP_ALL": 0.999})
    ok, info = check_promotion(
        tmp_path, "idea-d", 0.999,
        {"champion_guard": {"enabled": False}},
    )
    assert ok is True
    assert info["enabled"] is False


def test_history_persists_across_calls(tmp_path):
    for v in [0.88, 0.885, 0.89]:
        _write_metrics(tmp_path, f"idea-{v}", {"pgmAP_ALL": v})
        check_promotion(tmp_path, f"idea-{v}", v, {})
    hist = load_history(tmp_path)
    assert hist == [0.88, 0.885, 0.89]


def test_audit_idea_inserted_with_kind_audit(tmp_path):
    lake = IdeaLake(tmp_path / "l.db")
    create_audit_idea(lake, "audit-abc", "idea-susp",
                      {"claimed": 0.999, "verified": 0.91, "z": 5.5})
    row = lake.get("audit-abc")
    assert row["kind"] == "audit"
    assert row["parent"] == "idea-susp"
    assert row["status"] == "pending"


def test_reverify_reads_metrics_json_when_no_reproducer(tmp_path):
    _write_metrics(tmp_path, "idea-e", {"pgmAP_ALL": 0.75})
    # The claimed value is 0.9 but metrics.json says 0.75 — the guard
    # should use the verified 0.75 for z-score computation.
    save_history(tmp_path, [0.88] * 15)
    ok, info = check_promotion(tmp_path, "idea-e", 0.9, {})
    assert info["verified"] == 0.75
    # 0.75 vs mean 0.88 w/ std 0 gives z=None → allow (degenerate history).
    # Retest with real std
    save_history(tmp_path, [0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
                            0.87, 0.88, 0.89, 0.87])
    ok2, info2 = check_promotion(tmp_path, "idea-e", 0.9, {})
    assert info2["verified"] == 0.75
    # 0.75 is BELOW the mean so z is negative; should still allow.
    assert ok2 is True
