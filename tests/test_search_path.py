"""Tests for the search-path visualizer builder (orze.reporting.search_path).

Covers: graph/depth/subtree math, metric direction (asc vs desc), each problem
kind firing on a crafted fixture, cycle safety, empty input, and the config-
driven metric resolver.
"""

from orze.reporting.search_path import (
    Thresholds,
    build_search_path,
    make_metric_resolver,
    _config_delta,
    _classify_evolution,
)


def _metric_of(row):
    em = row.get("eval_metrics") or {}
    v = em.get("score")
    return float(v) if isinstance(v, (int, float)) else None


def _node(nodes, nid):
    return next(n for n in nodes if n["id"] == nid)


class TestStructure:
    def test_empty(self):
        d = build_search_path([], metric_of=lambda r: None, lower_is_better=True)
        assert d["nodes"] == [] and d["edges"] == []
        assert d["stats"]["n_total"] == 0

    def test_singletons_excluded(self):
        rows = [{"idea_id": "a", "parent": None}, {"idea_id": "b", "parent": None}]
        d = build_search_path(rows, metric_of=lambda r: None, lower_is_better=True)
        # no edges -> nothing is part of a tree
        assert d["nodes"] == []

    def test_depth_and_subtree(self):
        rows = [
            {"idea_id": "r", "parent": None},
            {"idea_id": "a", "parent": "r"},
            {"idea_id": "b", "parent": "a"},
            {"idea_id": "c", "parent": "r"},
        ]
        d = build_search_path(rows, metric_of=lambda r: None, lower_is_better=True)
        assert _node(d["nodes"], "r")["depth"] == 0
        assert _node(d["nodes"], "b")["depth"] == 2
        assert _node(d["nodes"], "r")["subtree_size"] == 4
        assert _node(d["nodes"], "a")["subtree_size"] == 2
        assert d["stats"]["max_depth"] == 2

    def test_all_nodes_positioned(self):
        rows = [
            {"idea_id": "r", "parent": None},
            {"idea_id": "a", "parent": "r"},
            {"idea_id": "b", "parent": "a"},
        ]
        d = build_search_path(rows, metric_of=lambda r: None, lower_is_better=True)
        assert all("x" in n and "y" in n for n in d["nodes"])


class TestMetricDirection:
    def _rows(self):
        return [
            {"idea_id": "r", "parent": None, "eval_metrics": {"score": 10.0}},
            {"idea_id": "c", "parent": "r", "eval_metrics": {"score": 5.0}},
        ]

    def test_lower_is_better_delta_positive_when_child_lower(self):
        d = build_search_path(self._rows(), metric_of=_metric_of,
                              lower_is_better=True)
        # child has lower (better) score -> positive (improving) delta
        assert _node(d["nodes"], "c")["delta_vs_parent"] > 0
        assert _node(d["nodes"], "c")["improved"] is True

    def test_higher_is_better_delta_negative_when_child_lower(self):
        d = build_search_path(self._rows(), metric_of=_metric_of,
                              lower_is_better=False)
        assert _node(d["nodes"], "c")["delta_vs_parent"] < 0
        assert _node(d["nodes"], "c")["improved"] is False


class TestProblems:
    def _kinds(self, problems):
        return {p["kind"] for p in problems}

    def test_under_researched(self):
        rows = [{"idea_id": "root", "parent": None,
                 "status": "completed", "eval_metrics": {"score": 1.0}}]
        # a strong completed leaf with no children, surrounded by weaker context
        rows += [{"idea_id": f"w{i}", "parent": "root",
                  "status": "completed", "eval_metrics": {"score": 100.0 + i}}
                 for i in range(10)]
        rows.append({"idea_id": "star", "parent": "root",
                     "status": "completed", "eval_metrics": {"score": 0.1}})
        d = build_search_path(rows, metric_of=_metric_of, lower_is_better=True)
        ur = [p for p in d["problems"] if p["kind"] == "under_researched"]
        assert any(p["node_id"] == "star" for p in ur)

    def test_saturated_parent_over_researched(self):
        # parent scored; 6 children, none beat the parent
        rows = [{"idea_id": "p", "parent": None, "status": "completed",
                 "eval_metrics": {"score": 1.0}}]
        rows += [{"idea_id": f"k{i}", "parent": "p", "status": "completed",
                  "eval_metrics": {"score": 5.0 + i}} for i in range(6)]
        d = build_search_path(rows, metric_of=_metric_of, lower_is_better=True)
        over = [p for p in d["problems"] if p["kind"] == "over_researched"]
        assert any(p["node_id"] == "p" for p in over)

    def test_flat_hub_no_evolution(self):
        # wide hub whose children are (almost) never evolved further
        rows = [{"idea_id": "hub", "parent": None, "status": "completed"}]
        rows += [{"idea_id": f"s{i}", "parent": "hub", "status": "completed"}
                 for i in range(20)]
        d = build_search_path(rows, metric_of=lambda r: None, lower_is_better=True)
        fh = [p for p in d["problems"] if p["kind"] == "flat_hub"]
        assert any(p["node_id"] == "hub" for p in fh)
        assert d["stats"]["evolution_rate"] == 0.0

    def test_evolved_hub_not_flat(self):
        # same fan-out but every child spawns a grandchild -> genuine evolution
        rows = [{"idea_id": "hub", "parent": None}]
        for i in range(20):
            rows.append({"idea_id": f"s{i}", "parent": "hub"})
            rows.append({"idea_id": f"g{i}", "parent": f"s{i}"})
        d = build_search_path(rows, metric_of=lambda r: None, lower_is_better=True)
        assert "flat_hub" not in self._kinds(d["problems"])
        assert d["stats"]["evolution_rate"] and d["stats"]["evolution_rate"] > 0.4

    def test_failed_cluster(self):
        rows = [{"idea_id": "p", "parent": None, "status": "failed"}]
        rows += [{"idea_id": f"f{i}", "parent": "p", "status": "failed"}
                 for i in range(4)]
        d = build_search_path(rows, metric_of=lambda r: None, lower_is_better=True)
        assert "failed_cluster" in self._kinds(d["problems"])

    def test_missing_coverage_flags_thin_bucket(self):
        rows = [{"idea_id": "r", "parent": None, "approach_family": "common"}]
        rows += [{"idea_id": f"c{i}", "parent": "r", "approach_family": "common"}
                 for i in range(20)]
        rows.append({"idea_id": "rare", "parent": "r", "approach_family": "rare"})
        d = build_search_path(rows, metric_of=lambda r: None, lower_is_better=True)
        mc = [p for p in d["problems"] if p["kind"] == "missing_coverage"]
        assert any("rare" in (p["region"] or "") for p in mc)

    def test_freetext_dimension_not_flagged(self):
        # many distinct categories => treated as free text, no coverage spam
        rows = [{"idea_id": "r", "parent": None, "category": "c0"}]
        rows += [{"idea_id": f"c{i}", "parent": "r", "category": f"cat{i}"}
                 for i in range(60)]
        th = Thresholds(coverage_max_buckets=40)
        d = build_search_path(rows, metric_of=lambda r: None,
                              lower_is_better=True, thresholds=th)
        flagged = [p for p in d["problems"]
                   if p["kind"] == "missing_coverage" and (p["region"] or "").startswith("category=")]
        assert flagged == []


class TestRobustness:
    def test_cycle_is_broken(self):
        rows = [
            {"idea_id": "a", "parent": "b", "status": "completed"},
            {"idea_id": "b", "parent": "a", "status": "completed"},
            {"idea_id": "r", "parent": None, "status": "completed"},
            {"idea_id": "c", "parent": "r", "status": "completed"},
        ]
        d = build_search_path(rows, metric_of=lambda r: None, lower_is_better=True)
        assert len(d["nodes"]) == 4
        assert all("x" in n for n in d["nodes"])
        # exactly one of the cycle pair becomes a root (parent dropped)
        roots = [n for n in d["nodes"] if n["parent"] is None]
        assert any(n["id"] in ("a", "b") for n in roots)

    def test_node_cap(self):
        # cap keeps whole trees (never splits a lineage); it drops *other* trees
        # once the budget is exhausted. Big tree (50) + small tree (3), cap 10.
        rows = [{"idea_id": "big", "parent": None}]
        rows += [{"idea_id": f"n{i}", "parent": "big"} for i in range(50)]
        rows += [{"idea_id": "small", "parent": None},
                 {"idea_id": "s1", "parent": "small"},
                 {"idea_id": "s2", "parent": "small"}]
        th = Thresholds(max_nodes=10)
        d = build_search_path(rows, metric_of=lambda r: None,
                              lower_is_better=True, thresholds=th)
        ids = {n["id"] for n in d["nodes"]}
        assert d["stats"]["truncated"] is True
        assert "big" in ids and "small" not in ids  # smaller tree dropped


class TestResolver:
    def test_direction_from_sort(self):
        cfg = {"report": {"primary_metric": "verified_wer", "sort": "ascending",
                          "columns": [{"key": "verified_wer",
                                       "source": "full_scale_metrics.json:avg_wer"}]}}
        metric_of, lower, name = make_metric_resolver(cfg)
        assert lower is True and name == "verified_wer"
        # resolves via the column source's json key
        assert metric_of({"eval_metrics": {"avg_wer": 5.4}}) == 5.4

    def test_descending_default(self):
        cfg = {"report": {"primary_metric": "score", "sort": "descending"}}
        metric_of, lower, name = make_metric_resolver(cfg)
        assert lower is False
        assert metric_of({"eval_metrics": {"score": 0.9}}) == 0.9

class TestConfigDelta:
    def test_single_key_change(self):
        d = _config_delta({"lr": 0.2, "bs": 8}, {"lr": 0.1, "bs": 8}, cap=16)
        assert d["size"] == 1
        assert d["changes"] == [{"key": "lr", "parent": 0.1, "child": 0.2}]

    def test_nested_flattening(self):
        d = _config_delta({"opt": {"lr": 0.2}}, {"opt": {"lr": 0.1}}, cap=16)
        assert d["size"] == 1 and d["changes"][0]["key"] == "opt.lr"

    def test_identical_is_zero(self):
        assert _config_delta({"a": 1}, {"a": 1}, cap=16)["size"] == 0

    def test_underscore_keys_ignored(self):
        # bookkeeping keys (e.g. _idea_id) must not count as a real change
        assert _config_delta({"_idea_id": "x", "a": 1}, {"a": 1}, cap=16)["size"] == 0

    def test_missing_config_is_undiffable(self):
        assert _config_delta(None, {"a": 1}, cap=16)["size"] == -1

    def test_cap_limits_reported_changes(self):
        child = {f"k{i}": i for i in range(20)}
        d = _config_delta(child, {}, cap=5)
        assert d["size"] == 20 and len(d["changes"]) == 5


class TestEvolutionClassify:
    def test_single_param(self):
        assert _classify_evolution(1, "train", "tweak lr", True) == "mutate_param"

    def test_multi_param(self):
        assert _classify_evolution(4, "train", "big change", True) == "multi_param"

    def test_zero_delta_is_replicate(self):
        assert _classify_evolution(0, "train", "duplicate config", True) == "replicate"

    def test_zero_delta_eval_keyword(self):
        assert _classify_evolution(0, "train", "cross-eval on French", True) == "cross_eval"

    def test_kind_combine(self):
        assert _classify_evolution(3, "bundle_combine", "merge views", True) == "combine"

    def test_undiffable_unknown(self):
        assert _classify_evolution(-1, "train", "x", True) == "unknown"


class TestEvolutionContract:
    def _rows(self):
        # parent with a config; one genuine child, one zero-delta child, one
        # changed-but-no-rationale child.
        return [
            {"idea_id": "p", "parent": None, "status": "completed",
             "config": {"lr": 0.1}, "hypothesis": "seed"},
            {"idea_id": "good", "parent": "p", "status": "completed",
             "config": {"lr": 0.2}, "hypothesis": "raise lr to escape plateau"},
            {"idea_id": "rerun", "parent": "p", "status": "completed",
             "config": {"lr": 0.1}, "hypothesis": "re-run"},
            {"idea_id": "nojust", "parent": "p", "status": "completed",
             "config": {"lr": 0.3}, "hypothesis": ""},
        ]

    def test_node_contract_fields(self):
        d = build_search_path(self._rows(), metric_of=lambda r: None,
                              lower_is_better=True)
        good = _node(d["nodes"], "good")
        assert good["contract_ok"] is True
        assert good["delta_size"] == 1
        assert good["evolution_type"] == "mutate_param"
        assert good["rationale"] == "raise lr to escape plateau"

    def test_zero_delta_flagged(self):
        d = build_search_path(self._rows(), metric_of=lambda r: None,
                              lower_is_better=True)
        rerun = _node(d["nodes"], "rerun")
        assert rerun["contract_ok"] is False
        assert "zero_delta" in rerun["contract_violations"]

    def test_no_rationale_flagged(self):
        d = build_search_path(self._rows(), metric_of=lambda r: None,
                              lower_is_better=True)
        nj = _node(d["nodes"], "nojust")
        assert nj["contract_ok"] is False
        assert "no_rationale" in nj["contract_violations"]

    def test_undiffable_contract_is_none(self):
        rows = [{"idea_id": "p", "parent": None, "config": {"a": 1}},
                {"idea_id": "c", "parent": "p", "config": None,
                 "hypothesis": "x"}]
        d = build_search_path(rows, metric_of=lambda r: None, lower_is_better=True)
        c = _node(d["nodes"], "c")
        assert c["contract_ok"] is None
        assert c["evolution_type"] == "unknown"

    def test_stats_genuine_rate_over_judged_only(self):
        d = build_search_path(self._rows(), metric_of=lambda r: None,
                              lower_is_better=True)
        s = d["stats"]
        assert s["n_edges"] == 3
        assert s["judged_edges"] == 3 and s["undiffable_edges"] == 0
        assert s["contract_ok_edges"] == 1
        assert s["zero_delta_edges"] == 1 and s["no_rationale_edges"] == 1
        assert abs(s["genuine_evolution_rate"] - 1 / 3) < 1e-3

    def test_pseudo_evolution_problem(self):
        # 3 zero-delta children of one parent -> pseudo_evolution flag
        rows = [{"idea_id": "p", "parent": None, "config": {"a": 1}}]
        rows += [{"idea_id": f"e{i}", "parent": "p", "config": {"a": 1},
                  "hypothesis": "eval"} for i in range(3)]
        th = Thresholds(eval_child_min=3)
        d = build_search_path(rows, metric_of=lambda r: None,
                              lower_is_better=True, thresholds=th)
        assert any(q["kind"] == "pseudo_evolution" for q in d["problems"])

    def test_unjustified_branch_problem(self):
        rows = [{"idea_id": "p", "parent": None, "config": {"a": 0}}]
        rows += [{"idea_id": f"c{i}", "parent": "p", "config": {"a": i + 1},
                  "hypothesis": ""} for i in range(3)]
        th = Thresholds(unjustified_min=3)
        d = build_search_path(rows, metric_of=lambda r: None,
                              lower_is_better=True, thresholds=th)
        assert any(q["kind"] == "unjustified_branch" for q in d["problems"])


from orze.reporting.search_path import compute_research_efficiency, _gini, _grade


class TestResearchEfficiency:
    def _eff(self, **kw):
        base = dict(
            n_total=100, n_scored=10, status_counts={"completed": 60, "failed": 40},
            fanout=[5, 3, 2], n_leaves=80, n_intermediate=20,
            refinement_success_rate=0.5, evolution_rate=0.3, depth_yield=[],
        )
        base.update(kw)
        return compute_research_efficiency(**base)

    def test_score_in_range_and_graded(self):
        e = self._eff()
        assert 0 <= e["score"] <= 100
        assert e["grade"] in ("A", "B", "C", "D", "F")

    def test_perfect_inputs_high_score(self):
        e = self._eff(n_scored=100, status_counts={"completed": 100},
                      fanout=[1, 1, 1, 1], refinement_success_rate=1.0,
                      evolution_rate=1.0)
        assert e["score"] >= 85 and e["grade"] == "A"

    def test_worst_inputs_low_score(self):
        e = self._eff(n_scored=0, status_counts={"failed": 100},
                      fanout=[100], refinement_success_rate=0.0,
                      evolution_rate=0.0)
        assert e["score"] <= 15 and e["grade"] == "F"

    def test_failure_rate_and_yield(self):
        e = self._eff()
        assert abs(e["failure_rate"] - 0.4) < 1e-6
        assert abs(e["yield_rate"] - 0.1) < 1e-6

    def test_concentration(self):
        e = self._eff(fanout=[8, 1, 1])  # 10 edges, top1 = 0.8
        assert abs(e["concentration"]["top1_share"] - 0.8) < 1e-6
        assert e["concentration"]["max_fanout"] == 8

    def test_exploit_share(self):
        e = self._eff(n_leaves=75, n_intermediate=25)
        assert abs(e["exploration_exploitation"]["exploit_share"] - 0.25) < 1e-6

    def test_weights_configurable(self):
        th = Thresholds(eff_w_reliability=10.0, eff_w_yield=0.0, eff_w_success=0.0,
                        eff_w_depth=0.0, eff_w_diversity=0.0)
        e = self._eff(th=th, status_counts={"completed": 100})  # reliability=1
        assert e["score"] == 100.0

    def test_gini_even_vs_concentrated(self):
        assert _gini([1, 1, 1, 1]) < 0.1
        assert _gini([0, 0, 0, 100]) > 0.6

    def test_grade_cutoffs(self):
        assert _grade(90) == "A" and _grade(60) == "C" and _grade(10) == "F"

    def test_efficiency_in_build_output(self):
        rows = [{"idea_id": "r", "parent": None, "status": "completed",
                 "eval_metrics": {"score": 1.0}}]
        rows += [{"idea_id": f"c{i}", "parent": "r", "status": "completed"} for i in range(3)]
        d = build_search_path(rows, metric_of=_metric_of, lower_is_better=False)
        re = d["research_efficiency"]
        assert "score" in re and "depth_yield" in re and "components" in re
