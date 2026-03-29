"""Tests for the auto-evolution feature stack.

Covers: failure_analysis, sealed, family_guard, retrospection dispatch,
        code_evolution context, meta_research stats, ideas approach_family,
        idea_lake migration.
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Feature 1: Structured Failure Analysis
# ---------------------------------------------------------------------------

class TestFailureAnalysis:
    def test_classify_oom(self):
        from orze.engine.failure_analysis import classify_failure
        assert classify_failure("CUDA out of memory", -1, "training") == "oom"
        assert classify_failure("OutOfMemoryError", -1, "training") == "oom"

    def test_classify_timeout(self):
        from orze.engine.failure_analysis import classify_failure
        assert classify_failure("Timed out", -1, "training") == "timeout"

    def test_classify_stall(self):
        from orze.engine.failure_analysis import classify_failure
        assert classify_failure("Stalled (no output for 30m)", -1, "training") == "stall"

    def test_classify_config_error(self):
        from orze.engine.failure_analysis import classify_failure
        assert classify_failure("KeyError: 'missing_key'", 1, "training") == "config_error"

    def test_classify_crash(self):
        from orze.engine.failure_analysis import classify_failure
        assert classify_failure("Segfault", 139, "training") == "crash"

    def test_classify_source_override(self):
        from orze.engine.failure_analysis import classify_failure
        assert classify_failure("anything", -1, "pre_script") == "pre_script_error"
        assert classify_failure("anything", -1, "eval") == "eval_failure"

    def test_build_failure_analysis(self):
        from orze.engine.failure_analysis import build_failure_analysis
        fa = build_failure_analysis("oom", "CUDA out of memory")
        assert fa["category"] == "oom"
        assert fa["what"] == "CUDA out of memory"
        assert "VRAM" in fa["why"]
        assert fa["lesson"]
        assert fa["timestamp"]

    def test_write_and_load(self, tmp_path):
        from orze.engine.failure_analysis import write_failure_analysis, load_recent_failures

        # Structured failure
        idea_dir = tmp_path / "idea-test01"
        idea_dir.mkdir()
        write_failure_analysis(idea_dir, "oom", "CUDA out of memory")
        assert (idea_dir / "failure_analysis.json").exists()

        # Legacy failure (metrics.json only)
        idea_dir2 = tmp_path / "idea-test02"
        idea_dir2.mkdir()
        (idea_dir2 / "metrics.json").write_text(json.dumps({
            "status": "FAILED", "error": "Timed out after 60m"
        }))

        result = load_recent_failures(tmp_path)
        assert "oom" in result
        assert "timeout" in result
        assert result["oom"][0]["idea_id"] == "idea-test01"


# ---------------------------------------------------------------------------
# Feature 2: Sealed Evaluation
# ---------------------------------------------------------------------------

class TestSealed:
    def test_hash_and_verify_clean(self, tmp_path):
        from orze.engine.sealed import compute_sealed_hashes, verify_sealed_files
        f = tmp_path / "eval.py"
        f.write_text("print(42)")
        hashes = compute_sealed_hashes([str(f)])
        assert str(f) in hashes
        assert len(hashes[str(f)]) == 64
        assert verify_sealed_files([str(f)], hashes) == []

    def test_hash_and_verify_tampered(self, tmp_path):
        from orze.engine.sealed import compute_sealed_hashes, verify_sealed_files
        f = tmp_path / "eval.py"
        f.write_text("print(42)")
        hashes = compute_sealed_hashes([str(f)])
        f.write_text("print(99)")
        assert verify_sealed_files([str(f)], hashes) == [str(f)]

    def test_manifest_roundtrip(self, tmp_path):
        from orze.engine.sealed import write_sealed_manifest, load_sealed_manifest
        hashes = {"a.py": "abc123"}
        write_sealed_manifest(tmp_path, hashes)
        assert load_sealed_manifest(tmp_path) == hashes

    def test_validate_metrics_ok(self):
        from orze.engine.sealed import validate_metrics
        ok, reason = validate_metrics({"status": "COMPLETED", "acc": 0.95}, {})
        assert ok and reason == ""

    def test_validate_metrics_nan(self):
        from orze.engine.sealed import validate_metrics
        ok, reason = validate_metrics({"status": "COMPLETED", "acc": float("nan")}, {})
        assert not ok and "NaN" in reason

    def test_validate_metrics_inf(self):
        from orze.engine.sealed import validate_metrics
        ok, reason = validate_metrics({"status": "COMPLETED", "loss": float("inf")}, {})
        assert not ok and "inf" in reason

    def test_validate_metrics_range(self):
        from orze.engine.sealed import validate_metrics
        ok, reason = validate_metrics(
            {"status": "COMPLETED", "acc": -0.5},
            {"metric_validation": {"min_value": {"acc": 0.0}}},
        )
        assert not ok and "below minimum" in reason


# ---------------------------------------------------------------------------
# Feature 3: Approach Family
# ---------------------------------------------------------------------------

class TestFamilyGuard:
    def test_infer_architecture(self):
        from orze.engine.family_guard import infer_approach_family
        assert infer_approach_family({"model": {"type": "resnet"}}, "") == "architecture"

    def test_infer_training_config(self):
        from orze.engine.family_guard import infer_approach_family
        assert infer_approach_family({"lr": 0.001, "optimizer": "adam"}, "") == "training_config"

    def test_infer_data(self):
        from orze.engine.family_guard import infer_approach_family
        assert infer_approach_family({"augment": True, "data": {}}, "") == "data"

    def test_infer_other(self):
        from orze.engine.family_guard import infer_approach_family
        assert infer_approach_family({"foo": "bar"}, "") == "other"

    def test_concentration_detected(self):
        from orze.engine.family_guard import check_family_concentration
        assert check_family_concentration(["architecture"] * 5, 5) is not None

    def test_concentration_ok(self):
        from orze.engine.family_guard import check_family_concentration
        assert check_family_concentration(["architecture"] * 4 + ["data"], 5) is None

    def test_concentration_too_few(self):
        from orze.engine.family_guard import check_family_concentration
        assert check_family_concentration(["architecture"] * 3, 5) is None


class TestIdeasApproachFamily:
    def test_parse_with_family(self, tmp_path):
        from orze.core.ideas import parse_ideas
        f = tmp_path / "ideas.md"
        f.write_text(
            "## idea-aaa111: Test Idea\n"
            "- **Priority**: high\n"
            "- **Approach Family**: training_config\n"
            "- **Config overrides**:\n"
            "  ```yaml\n  lr: 0.001\n  ```\n"
        )
        ideas = parse_ideas(str(f))
        assert ideas["idea-aaa111"]["approach_family"] == "training_config"

    def test_parse_default_family(self, tmp_path):
        from orze.core.ideas import parse_ideas
        f = tmp_path / "ideas.md"
        f.write_text(
            "## idea-bbb222: No Family\n"
            "- **Priority**: medium\n"
            "- **Config overrides**:\n"
            "  ```yaml\n  x: 1\n  ```\n"
        )
        ideas = parse_ideas(str(f))
        assert ideas["idea-bbb222"]["approach_family"] == "other"


class TestIdeaLakeMigration:
    def test_migration_adds_column(self, tmp_path):
        """Old DB without approach_family gets it via migration."""
        from orze.idea_lake import IdeaLake

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE ideas (
            idea_id TEXT PRIMARY KEY, id_num INTEGER, title TEXT NOT NULL,
            priority TEXT DEFAULT 'medium', category TEXT DEFAULT 'architecture',
            parent TEXT, hypothesis TEXT, config TEXT NOT NULL,
            raw_markdown TEXT NOT NULL, config_summary TEXT, eval_metrics TEXT,
            status TEXT DEFAULT 'archived', training_time REAL,
            archived_at TEXT, created_at TEXT
        )""")
        conn.execute("CREATE TABLE id_sequence (next_id INTEGER NOT NULL)")
        conn.execute("INSERT INTO id_sequence VALUES (1)")
        conn.execute(
            "INSERT INTO ideas (idea_id, id_num, title, config, raw_markdown, status) "
            "VALUES ('idea-000001', 1, 'Old', 'lr: 0.1', 'raw', 'completed')"
        )
        conn.commit()
        conn.close()

        lake = IdeaLake(db_path)
        cols = {r[1] for r in lake.conn.execute("PRAGMA table_info(ideas)").fetchall()}
        assert "approach_family" in cols

        row = lake.conn.execute(
            "SELECT approach_family FROM ideas WHERE idea_id = 'idea-000001'"
        ).fetchone()
        assert row[0] == "other"

        lake.insert("idea-000002", "New", "lr: 0.01", "raw",
                     status="queued", approach_family="architecture")
        row = lake.conn.execute(
            "SELECT approach_family FROM ideas WHERE idea_id = 'idea-000002'"
        ).fetchone()
        assert row[0] == "architecture"
        lake.close()


# ---------------------------------------------------------------------------
# Feature 5: Retrospection Dispatch
# ---------------------------------------------------------------------------

class TestRetrospectionDispatch:
    def test_classify_signals(self):
        from orze.engine.retrospection import _classify_signals
        assert "plateau" in _classify_signals("PLATEAU: no improvement")
        assert "high_failure_rate" in _classify_signals("HIGH FAILURE RATE: 60%")
        assert "family_imbalance" in _classify_signals("FAMILY CONCENTRATION: x")
        assert _classify_signals("PLATEAU: x; HIGH FAILURE RATE: y") == [
            "plateau", "high_failure_rate"
        ]
        assert _classify_signals("unknown") == ["persistent_failure"]

    def test_dispatch_with_evolution(self, tmp_path):
        from orze.engine.retrospection import _dispatch_signal, is_research_paused

        cfg = {"evolution": {"enabled": True},
               "retrospection": {"evolution_attempts_before_pause": 2}}
        state = {}

        # First attempt → triggers code_evolution
        action = _dispatch_signal("plateau", tmp_path, cfg, state)
        assert action == "code_evolution"
        assert (tmp_path / "_trigger_code_evolution").exists()
        assert state["plateau_evolution_attempts"] == 1
        (tmp_path / "_trigger_code_evolution").unlink()

        # Second attempt → still code_evolution
        action = _dispatch_signal("plateau", tmp_path, cfg, state)
        assert action == "code_evolution"
        assert state["plateau_evolution_attempts"] == 2
        (tmp_path / "_trigger_code_evolution").unlink()

        # Third attempt → exhausted → pause
        action = _dispatch_signal("plateau", tmp_path, cfg, state)
        assert action == "pause"
        assert is_research_paused(tmp_path)

    def test_dispatch_evolution_disabled(self, tmp_path):
        from orze.engine.retrospection import _dispatch_signal, is_research_paused

        cfg = {"evolution": {"enabled": False}}
        action = _dispatch_signal("plateau", tmp_path, cfg, {})
        assert action == "pause"
        assert is_research_paused(tmp_path)

    def test_dispatch_family_imbalance(self, tmp_path):
        from orze.engine.retrospection import _dispatch_signal

        cfg = {"evolution": {"enabled": True},
               "retrospection": {"evolution_attempts_before_pause": 2}}
        action = _dispatch_signal("family_imbalance", tmp_path, cfg, {})
        assert action == "meta_research"
        assert (tmp_path / "_trigger_meta_research").exists()


# ---------------------------------------------------------------------------
# Feature 4: Code Evolution (context building only — no Claude CLI)
# ---------------------------------------------------------------------------

class TestCodeEvolution:
    def test_build_context(self, tmp_path):
        from orze.agents.code_evolution import build_evolution_context
        results = tmp_path / "results"
        results.mkdir()
        ideas = tmp_path / "ideas.md"
        ideas.write_text("# Ideas")
        ctx = build_evolution_context(results, ideas, {"report": {"primary_metric": "acc"}})
        assert "Code Evolution Context" in ctx

    def test_build_prompt(self):
        from orze.agents.code_evolution import build_evolution_prompt
        prompt = build_evolution_prompt("ctx", "plateau", "print(42)", ["eval.py"])
        assert "plateau" in prompt
        assert "print(42)" in prompt
        assert "eval.py" in prompt
        assert "Backward compatible" in prompt


# ---------------------------------------------------------------------------
# Feature 5: Meta-Research (prompt building only — no Claude CLI)
# ---------------------------------------------------------------------------

class TestMetaResearch:
    def test_build_strategy_prompt(self):
        from orze.agents.meta_research import build_strategy_prompt
        stats = {
            "architecture": {"count": 50, "share": 0.25, "mean": 0.85, "best": 0.92},
            "training_config": {"count": 100, "share": 0.50, "mean": 0.80, "best": 0.88},
        }
        prompt = build_strategy_prompt(stats, "rules here", "family_imbalance")
        assert "family_imbalance" in prompt
        assert "architecture" in prompt
        assert "50%" in prompt
        assert "rules here" in prompt
