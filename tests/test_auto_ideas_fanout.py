"""Fan-out cap policy in the free research loop (research-efficiency lever).

The free loop must stop piling every variation onto one champion hub; once a
parent hits the fan-out cap, selection should skip it and pick the next-best
under-saturated candidate, which spreads search and deepens lineages.
"""
import json

import yaml

from orze.engine.auto_ideas import _get_best_config
from orze.idea_lake import IdeaLake


def _make_idea(results_dir, name, primary, cfg=None):
    d = results_dir / name
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps(
        {"status": "COMPLETED", "verified_wer": primary}))
    (d / "idea_config.yaml").write_text(yaml.dump(cfg or {"learning_rate": 1e-4}))


def _cfg():
    return {"report": {"primary_metric": "verified_wer", "sort": "ascending"}}


class TestFanoutCap:
    def test_picks_global_best_when_uncapped(self, tmp_path):
        _make_idea(tmp_path, "idea-aaa", 5.0)
        _make_idea(tmp_path, "idea-bbb", 4.0)  # best (lower WER)
        best = _get_best_config(tmp_path, _cfg())
        assert best["_idea_id"] == "idea-bbb"

    def test_skips_saturated_hub(self, tmp_path):
        _make_idea(tmp_path, "idea-aaa", 5.0)
        _make_idea(tmp_path, "idea-bbb", 4.0)  # best but saturated
        best = _get_best_config(
            tmp_path, _cfg(),
            child_counts={"idea-bbb": 12}, fanout_cap=12)
        # bbb is capped out -> falls through to next-best aaa
        assert best["_idea_id"] == "idea-aaa"

    def test_under_cap_still_selected(self, tmp_path):
        _make_idea(tmp_path, "idea-bbb", 4.0)
        best = _get_best_config(
            tmp_path, _cfg(),
            child_counts={"idea-bbb": 11}, fanout_cap=12)
        assert best["_idea_id"] == "idea-bbb"

    def test_zero_cap_disables(self, tmp_path):
        _make_idea(tmp_path, "idea-bbb", 4.0)
        best = _get_best_config(
            tmp_path, _cfg(),
            child_counts={"idea-bbb": 999}, fanout_cap=0)
        assert best["_idea_id"] == "idea-bbb"


class TestChildCounts:
    def test_counts_children_per_parent(self, tmp_path):
        lake = IdeaLake(str(tmp_path / "lake.db"))
        lake.insert("idea-root", "root", "{}", "raw")
        for i in range(3):
            lake.insert(f"idea-c{i}", f"c{i}", "{}", "raw",
                        parent="idea-root", hypothesis="because")
        lake.insert("idea-solo", "solo", "{}", "raw",
                    parent="idea-other", hypothesis="x")
        counts = lake.child_counts()
        assert counts.get("idea-root") == 3
        assert counts.get("idea-other") == 1
        assert "idea-root" not in [k for k, v in counts.items() if v == 0]

    def test_ignores_none_parents(self, tmp_path):
        lake = IdeaLake(str(tmp_path / "lake.db"))
        lake.insert("idea-a", "a", "{}", "raw", parent="none")
        lake.insert("idea-b", "b", "{}", "raw", parent="")
        lake.insert("idea-c", "c", "{}", "raw")
        assert lake.child_counts() == {}
