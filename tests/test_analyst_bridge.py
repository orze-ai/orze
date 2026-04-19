"""Tests for F7 — analyst bridge agent."""
import json
from pathlib import Path

from orze.agents import analyst_bridge as ab


def test_extract_actionable_filters_noise():
    text = """
# Analyst Insights

## Quick wins
- Immediate quick win: Switch from LQM to late_k2 for mAP=0.890 (+0.005). Zero training needed.
- Rescue ceiling: Using max gives AP=0.895 (+0.011). One-line change.

## Notes
This paragraph is just context without any concrete action.
Multi-model oracle = AP 0.999. No training needed for this adjustment.
"""
    actions = ab.extract_actionable(text)
    # Each actionable hit should be represented exactly once.
    assert any("+0.005" in a for a in actions)
    assert any("+0.011" in a for a in actions)
    # Pure context without action-signal is excluded.
    assert not any("just context" in a for a in actions)


def test_run_emits_ideas_first_time(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "_analyst_insights.md").write_text(
        "- Immediate quick win: swap LQM for late_k2 (+0.005 mAP, zero training).\n"
        "- Rescue with max for 44 samples gives +0.011 mAP, no training.\n",
        encoding="utf-8")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n\n", encoding="utf-8")

    summary = ab.run(results, ideas)
    assert summary["emitted"] >= 2
    body = ideas.read_text(encoding="utf-8")
    assert "origin:analyst" in body
    assert "idea-analyst-" in body


def test_run_is_idempotent(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "_analyst_insights.md").write_text(
        "- Switch LQM to late_k2 for +0.005 mAP, zero training.\n",
        encoding="utf-8")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n\n", encoding="utf-8")

    s1 = ab.run(results, ideas)
    assert s1["emitted"] == 1
    s2 = ab.run(results, ideas)
    assert s2["emitted"] == 0, "second run should not re-emit"
    # Ideas file should still contain exactly one analyst idea.
    assert ideas.read_text().count("idea-analyst-") == 1


def test_run_no_source_files(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    ideas = tmp_path / "ideas.md"
    s = ab.run(results, ideas)
    assert s["emitted"] == 0
    assert s["reason"] == "no-source-files"
