"""Search Path Visualizer — research genealogy graph + problem detection.

This module is **task-agnostic**. It knows nothing about any specific domain,
dataset, or metric. The caller supplies:

  * ``rows``        — one dict per idea with at least ``idea_id`` and ``parent``;
                      optional ``title``, ``category``, ``approach_family``,
                      ``status``, ``priority``, ``created_at``, ``training_time``.
  * ``metric_of``   — callable ``row -> Optional[float]`` returning the idea's
                      primary metric (whatever that is for the project), or
                      ``None`` when the idea has no comparable score.
  * ``lower_is_better`` — metric direction (e.g. ``report.sort == 'ascending'``).

``build_search_path`` returns a JSON-serialisable dict describing the genealogy
forest (nodes + edges with server-computed tidy-tree layout), a ranked list of
``problems`` (under-/over-researched, failed clusters, missing coverage), and a
taxonomy ``coverage`` summary. The goal is that a human can glance at the graph
and the attention list and immediately find the parts of the search that are
broken, neglected, or wasteful.

CLI:
    python -m orze.reporting.search_path --db .orze/idea_lake.db \\
        [--config orze.yaml] [--json out.json]
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

# Statuses treated as "this idea actually ran and produced a result we trust".
_SCORED_STATUSES = {"completed", "partial"}
# Statuses that represent wasted / broken compute.
_FAILED_STATUSES = {"failed", "error"}

# Layout spacing (abstract units; the frontend scales as needed).
_X_GAP = 1.0
_Y_GAP = 1.0
_TREE_GAP = 2.0


@dataclass
class Thresholds:
    """Tunable knobs for problem detection. All dimensionless / project-neutral."""

    under_top_quantile: float = 0.20   # leaf in top 20% of scores w/ no children
    over_min_chain: int = 4            # refinement depth before "too deep" applies
    over_stagnant_run: int = 3         # consecutive non-improving descendants
    saturated_fanout: int = 5          # children of one parent before "saturated"
    flat_hub_fanout: int = 12          # children before "is this hub evolving?"
    flat_hub_evolve_ratio: float = 0.05  # frac of children that spawn grandchildren
    fail_cluster_min_size: int = 3     # min subtree size to call a failed cluster
    fail_cluster_frac: float = 0.60    # fraction failed to flag the cluster
    coverage_thin_ratio: float = 0.25  # bucket < ratio*mean -> under-explored
    coverage_max_buckets: int = 40     # dims with more distinct values = free-text
    coverage_max_flags: int = 8        # cap thin-bucket flags per dimension
    eval_child_min: int = 3            # zero-delta children before "eval mislabeled"
    unjustified_min: int = 3           # children w/o rationale before flagging a hub
    eff_w_yield: float = 1.0       # scored ideas per idea produced
    eff_w_success: float = 1.5     # refinements that actually improve
    eff_w_depth: float = 1.5       # search compounds (depth) vs flat breadth
    eff_w_diversity: float = 1.0   # not all compute on one champion arm
    eff_w_reliability: float = 1.0 # runs succeed rather than error/fail
    eff_yield_target: float = 0.10 # scored/total at/above this == full yield credit
    delta_report_cap: int = 16         # max changed keys reported per edge
    max_nodes: int = 4000              # safety cap on rendered nodes

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "Thresholds":
        sp = ((cfg or {}).get("report", {}) or {}).get("search_path", {}) or {}
        base = cls()
        for f in base.__dataclass_fields__:  # type: ignore[attr-defined]
            if f in sp:
                setattr(base, f, sp[f])
        return base


def _goodness(metric: Optional[float], lower_is_better: bool) -> Optional[float]:
    """Map a raw metric onto a 'higher is better' axis so comparisons are uniform."""
    if metric is None or (isinstance(metric, float) and math.isnan(metric)):
        return None
    return -float(metric) if lower_is_better else float(metric)


# ---------------------------------------------------------------------------
#  Evolution contract: what makes an edge a *genuine* evolution.
#
#  An edge child -> parent qualifies as an evolution iff BOTH hold:
#    1. a non-empty config delta (something concrete changed vs the parent), and
#    2. a non-empty rationale (a stated reason the change should help).
#  These checks are task-agnostic: the config is diffed structurally and the
#  rationale is whatever free-text justification the project records per idea.
# ---------------------------------------------------------------------------

# Generic evolution operators. Inferred from the config delta + idea kind/title;
# never domain-specific.
EVOLUTION_TYPES = (
    "seed",          # a root: no parent, the start of a lineage
    "mutate_param",  # exactly one config key changed
    "multi_param",   # several config keys changed at once
    "combine",       # merges/combines configs or views (kind/title signalled)
    "cross_eval",    # re-uses the parent's config to evaluate elsewhere
    "audit",         # re-tests a parent (champion-promotion audit etc.)
    "replicate",     # identical config to parent (a re-run / no real change)
    "unknown",       # parented but config unavailable to diff
)

_SCALAR = (str, int, float, bool, type(None))


def _flatten_config(cfg: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a (possibly nested) config dict to dotted scalar keys.

    Lists and non-dict containers are compared as a whole (stringified) so a
    reordering or element change registers as a single key delta.
    """
    out: Dict[str, Any] = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(_flatten_config(v, key + "."))
            elif isinstance(v, _SCALAR):
                out[key] = v
            else:
                out[key] = json.dumps(v, sort_keys=True, default=str)
    elif prefix:
        out[prefix.rstrip(".")] = cfg
    return out


def _config_delta(child: Any, parent: Any, cap: int) -> Dict[str, Any]:
    """Structural diff of two configs.

    Returns ``{"changes": [...], "size": N}`` where each change is
    ``{"key", "parent", "child"}``. ``size`` is the *total* number of differing
    keys; ``changes`` is capped to keep the payload bounded. Keys whose name
    starts with ``_`` (bookkeeping such as ``_idea_id``) are ignored.
    """
    if not isinstance(child, dict) or not isinstance(parent, dict):
        return {"changes": [], "size": -1}  # -1 = undiffable (config missing)
    fc = {k: v for k, v in _flatten_config(child).items() if not k.split(".")[-1].startswith("_")}
    fp = {k: v for k, v in _flatten_config(parent).items() if not k.split(".")[-1].startswith("_")}
    changes: List[dict] = []
    for k in sorted(set(fc) | set(fp)):
        cv, pv = fc.get(k, "—"), fp.get(k, "—")
        if cv != pv:
            if len(changes) < cap:
                changes.append({"key": k, "parent": pv, "child": cv})
    size = sum(1 for k in set(fc) | set(fp) if fc.get(k, "—") != fp.get(k, "—"))
    return {"changes": changes, "size": size}


def _classify_evolution(delta_size: int, kind: Optional[str], title: str,
                        has_rationale: bool) -> str:
    """Assign a generic evolution operator to an edge from cheap signals."""
    k = (kind or "").lower()
    if k in ("bundle_combine", "agg_search"):
        return "combine"
    if k == "audit":
        return "audit"
    t = title.lower()
    if delta_size == 0:
        # Same config as the parent. A keyword hints whether it was a
        # deliberate re-evaluation vs an accidental duplicate.
        if any(w in t for w in ("eval", "cross", "ablation", "verify", "re-run", "rerun")):
            return "cross_eval"
        return "replicate"
    if delta_size < 0:
        return "unknown"
    if any(w in t for w in ("combine", "merge", "ensemble", "all best", "best settings")):
        return "combine"
    return "mutate_param" if delta_size == 1 else "multi_param"


def _gini(values: Sequence[float]) -> float:
    """Gini coefficient of a non-negative distribution (0 = even, 1 = concentrated)."""
    xs = sorted(float(v) for v in values if v is not None)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return round((2 * cum) / (n * sum(xs)) - (n + 1) / n, 4)


def _grade(score: float) -> str:
    for cut, g in ((85, "A"), (70, "B"), (55, "C"), (40, "D")):
        if score >= cut:
            return g
    return "F"


def compute_research_efficiency(
    *,
    n_total: int,
    n_scored: int,
    status_counts: Dict[str, int],
    fanout: Sequence[int],
    n_leaves: int,
    n_intermediate: int,
    refinement_success_rate: Optional[float],
    evolution_rate: Optional[float],
    depth_yield: List[dict],
    th: Optional["Thresholds"] = None,
) -> Dict[str, Any]:
    """Composite, task-agnostic *research-efficiency* score for the whole run.

    Research efficiency answers "is the search producing results per unit of
    research effort, and is it compounding?" — orze's own meta-metric for how
    well it *researches*, distinct from compute/GPU efficiency and from the
    project's task metric. All sub-scores are normalised to 0..1 (higher =
    better) and blended with configurable weights into a 0..100 score + grade.
    """
    th = th or Thresholds()
    total = max(n_total, 1)

    # reliability: share of attempts that were not failures/errors.
    failed = sum(v for k, v in status_counts.items()
                 if k.lower() in _FAILED_STATUSES)
    failure_rate = failed / total
    reliability = 1.0 - failure_rate

    # yield: scored results per idea produced, saturating at a target rate.
    yield_rate = n_scored / total
    yield_score = min(1.0, yield_rate / th.eff_yield_target) if th.eff_yield_target else 0.0

    # depth utilisation: how often a parented idea is itself evolved further.
    depth_util = float(evolution_rate or 0.0)

    # success: refinements that improved on their parent.
    success = float(refinement_success_rate or 0.0)

    # diversity: 1 - share of edges held by the single biggest hub.
    n_edges = sum(fanout)
    top1 = max(fanout) if fanout else 0
    top1_share = top1 / n_edges if n_edges else 0.0
    sorted_fan = sorted(fanout, reverse=True)
    top5_share = (sum(sorted_fan[:5]) / n_edges) if n_edges else 0.0
    diversity = 1.0 - top1_share

    comps = {
        "yield": {"value": round(yield_rate, 4), "score": round(yield_score, 4),
                  "weight": th.eff_w_yield},
        "success": {"value": round(success, 4), "score": round(success, 4),
                    "weight": th.eff_w_success},
        "depth_utilization": {"value": round(depth_util, 4), "score": round(depth_util, 4),
                              "weight": th.eff_w_depth},
        "diversity": {"value": round(diversity, 4), "score": round(diversity, 4),
                      "weight": th.eff_w_diversity},
        "reliability": {"value": round(reliability, 4), "score": round(reliability, 4),
                        "weight": th.eff_w_reliability},
    }
    wsum = sum(c["weight"] for c in comps.values()) or 1.0
    score = 100.0 * sum(c["score"] * c["weight"] for c in comps.values()) / wsum
    score = round(score, 1)

    explore = n_leaves
    exploit = n_intermediate
    eo_total = explore + exploit
    return {
        "score": score,
        "grade": _grade(score),
        "components": comps,
        "weights_sum": wsum,
        "exploration_exploitation": {
            "explore": explore,
            "exploit": exploit,
            "exploit_share": round(exploit / eo_total, 4) if eo_total else 0.0,
        },
        "concentration": {
            "top1_share": round(top1_share, 4),
            "top5_share": round(top5_share, 4),
            "max_fanout": top1,
            "gini": _gini(fanout),
        },
        "failure_rate": round(failure_rate, 4),
        "yield_rate": round(yield_rate, 4),
        "depth_yield": depth_yield,
    }


def build_search_path(
    rows: Sequence[dict],
    *,
    metric_of: Callable[[dict], Optional[float]],
    lower_is_better: bool,
    metric_name: str = "metric",
    thresholds: Optional[Thresholds] = None,
) -> Dict[str, Any]:
    """Build the genealogy graph and problem annotations. Pure / no I/O."""
    th = thresholds or Thresholds()

    # ---- index rows, resolve metrics ----
    by_id: Dict[str, dict] = {}
    for r in rows:
        iid = r.get("idea_id")
        if iid:
            by_id[iid] = r

    parent_of: Dict[str, str] = {}
    children: Dict[str, List[str]] = defaultdict(list)
    for iid, r in by_id.items():
        p = r.get("parent")
        if p and p not in ("none", "") and p != iid and p in by_id:
            parent_of[iid] = p
            children[p].append(iid)

    # Guarantee a forest: if a parent chain loops back on itself (corrupt data),
    # cut the edge that closes the cycle so every node is reachable from a root.
    for iid in list(parent_of):
        seen = {iid}
        cur = parent_of.get(iid)
        while cur is not None:
            if cur in seen:
                p = parent_of.pop(iid)
                children[p].remove(iid)
                break
            seen.add(cur)
            cur = parent_of.get(cur)

    # Restrict the visual graph to the genealogy forest (drop isolated singletons:
    # ideas with neither a parent nor any children carry no path information).
    in_tree = set(parent_of) | set(children)

    goodness: Dict[str, Optional[float]] = {}
    raw_metric: Dict[str, Optional[float]] = {}
    for iid, r in by_id.items():
        m = metric_of(r)
        raw_metric[iid] = m
        goodness[iid] = _goodness(m, lower_is_better)

    # ---- percentile rank among scored nodes (1.0 == best) ----
    scored = sorted(
        (iid for iid in by_id if goodness[iid] is not None),
        key=lambda i: goodness[i],  # type: ignore[arg-type]
    )
    pct: Dict[str, float] = {}
    n_scored = len(scored)
    for rank, iid in enumerate(scored):
        pct[iid] = (rank + 1) / n_scored if n_scored else 0.0

    # ---- depth (cycle-safe) ----
    depth: Dict[str, int] = {}

    def _depth(iid: str, seen: Optional[set] = None) -> int:
        if iid in depth:
            return depth[iid]
        seen = seen or set()
        if iid in seen:
            return 0
        seen.add(iid)
        p = parent_of.get(iid)
        d = 0 if p is None else _depth(p, seen) + 1
        depth[iid] = d
        return d

    for iid in in_tree:
        _depth(iid)

    roots = sorted(
        (iid for iid in in_tree if iid not in parent_of),
        key=lambda i: -_subtree_size(i, children),
    )

    # ---- per-subtree aggregates (size, failed fraction, best goodness, depth) ----
    subtree_size: Dict[str, int] = {}
    failed_count: Dict[str, int] = {}
    best_good: Dict[str, Optional[float]] = {}
    subtree_depth: Dict[str, int] = {}

    def _agg(iid: str) -> None:
        size = 1
        failed = 1 if (by_id[iid].get("status") or "").lower() in _FAILED_STATUSES else 0
        best = goodness[iid]
        sdepth = 0
        for c in children.get(iid, ()):  # post-order
            _agg(c)
            size += subtree_size[c]
            failed += failed_count[c]
            sdepth = max(sdepth, subtree_depth[c] + 1)
            cb = best_good[c]
            if cb is not None and (best is None or cb > best):
                best = cb
        subtree_size[iid] = size
        failed_count[iid] = failed
        best_good[iid] = best
        subtree_depth[iid] = sdepth

    for r in roots:
        _agg(r)

    # ---- best ancestor goodness + non-improving run length (for "over-researched") ----
    best_ancestor: Dict[str, Optional[float]] = {}
    stagnant_run: Dict[str, int] = {}

    def _ancestor(iid: str) -> None:
        p = parent_of.get(iid)
        if p is None:
            best_ancestor[iid] = None
            stagnant_run[iid] = 0
        else:
            pa = best_ancestor[p]
            pg = goodness[p]
            ba = pg if (pa is None or (pg is not None and pg > pa)) else pa
            best_ancestor[iid] = ba
            g = goodness[iid]
            improved = g is not None and ba is not None and g > ba
            no_signal = g is None or ba is None
            # A "stagnant" step is one that had a comparable score but did not beat
            # the best result already seen on its lineage.
            stagnant_run[iid] = 0 if (improved or no_signal) else stagnant_run[p] + 1
        for c in children.get(iid, ()):
            _ancestor(c)

    for r in roots:
        _ancestor(r)

    # ---- tidy-tree layout (server-side; O(n)) ----
    x_pos: Dict[str, float] = {}
    y_pos: Dict[str, float] = {}
    cursor = [0.0]

    def _layout(iid: str) -> None:
        kids = sorted(children.get(iid, ()), key=lambda c: by_id[c].get("created_at") or c)
        if not kids:
            x_pos[iid] = cursor[0]
            cursor[0] += _X_GAP
        else:
            for c in kids:
                _layout(c)
            x_pos[iid] = (x_pos[kids[0]] + x_pos[kids[-1]]) / 2.0
        y_pos[iid] = depth[iid] * _Y_GAP

    for r in roots:
        _layout(r)
        cursor[0] += _TREE_GAP

    # ---- evolution classification (per edge) ----
    # For every child, diff its config against its parent's, attach the stated
    # rationale, infer the operator, and evaluate the evolution contract.
    evo: Dict[str, dict] = {}
    for iid in in_tree:
        p = parent_of.get(iid)
        if p is None:
            evo[iid] = {
                "evolution_type": "seed", "parent_delta": [], "delta_size": 0,
                "rationale": (by_id[iid].get("rationale")
                              or by_id[iid].get("hypothesis") or "").strip() or None,
                "contract_ok": True, "contract_violations": [],
            }
            continue
        d = _config_delta(by_id[iid].get("config"), by_id[p].get("config"),
                          th.delta_report_cap)
        rationale = (by_id[iid].get("rationale")
                     or by_id[iid].get("hypothesis") or "").strip()
        etype = _classify_evolution(d["size"], by_id[iid].get("kind"),
                                    by_id[iid].get("title") or iid, bool(rationale))
        violations: List[str] = []
        if d["size"] == 0:
            violations.append("zero_delta")      # identical config -> not an evolution
        if not rationale:
            violations.append("no_rationale")    # change with no stated justification
        if d["size"] < 0:
            # Config unavailable on one side: the delta half of the contract can't
            # be judged. contract_ok is None (unknown), not a pass or a fail.
            contract_ok: Optional[bool] = None
        else:
            contract_ok = d["size"] != 0 and bool(rationale)
        evo[iid] = {
            "evolution_type": etype,
            "parent_delta": d["changes"],
            "delta_size": d["size"],
            "rationale": rationale or None,
            "contract_ok": contract_ok,
            "contract_violations": violations,
        }

    # ---- assemble nodes ----
    def _delta(iid: str) -> Optional[float]:
        p = parent_of.get(iid)
        if p is None:
            return None
        g, gp = goodness[iid], goodness[p]
        if g is None or gp is None:
            return None
        return round(g - gp, 6)

    nodes: List[dict] = []
    node_problems: Dict[str, List[str]] = defaultdict(list)
    for iid in in_tree:
        r = by_id[iid]
        nodes.append({
            "id": iid,
            "title": r.get("title") or iid,
            "parent": parent_of.get(iid),
            "category": (r.get("category") or "unknown"),
            "approach_family": (r.get("approach_family") or "other"),
            "status": (r.get("status") or "unknown"),
            "priority": r.get("priority"),
            "metric": raw_metric[iid],
            "score_pct": round(pct.get(iid, 0.0), 4) if iid in pct else None,
            "depth": depth[iid],
            "n_children": len(children.get(iid, ())),
            "subtree_size": subtree_size.get(iid, 1),
            "subtree_depth": subtree_depth.get(iid, 0),
            "delta_vs_parent": _delta(iid),
            "improved": (_delta(iid) or 0) > 0,
            "evolution_type": evo[iid]["evolution_type"],
            "parent_delta": evo[iid]["parent_delta"],
            "delta_size": evo[iid]["delta_size"],
            "rationale": evo[iid]["rationale"],
            "contract_ok": evo[iid]["contract_ok"],
            "contract_violations": evo[iid]["contract_violations"],
            "training_time": r.get("training_time"),
            "x": round(x_pos[iid], 4),
            "y": round(y_pos[iid], 4),
            "problems": node_problems[iid],  # filled below (same list object)
        })
    node_index = {n["id"]: n for n in nodes}

    # ---- problem detection ----
    problems: List[dict] = []

    def _add(kind, severity, reason, suggestion, node_id=None, region=None):
        problems.append({
            "kind": kind, "severity": severity, "reason": reason,
            "suggestion": suggestion, "node_id": node_id, "region": region,
        })
        if node_id is not None:
            node_problems[node_id].append(kind)

    top_cut = 1.0 - th.under_top_quantile
    for iid in in_tree:
        r = by_id[iid]
        status = (r.get("status") or "").lower()
        n_kids = len(children.get(iid, ()))
        p = pct.get(iid)

        # under-researched: a strong result nobody refined
        if (status in _SCORED_STATUSES and n_kids == 0 and p is not None and p >= top_cut):
            sev = "high" if p >= 0.95 else "medium"
            _add("under_researched", sev,
                 f"Top-{max(1, int(round((1 - p) * 100)))}% result but has no follow-up ideas.",
                 f"Spawn child ideas refining {iid}.", node_id=iid)

        # flat hub: a parent with wide fan-out whose children are almost never
        # evolved further — breadth-first sibling spawning instead of evolution.
        if n_kids >= th.flat_hub_fanout:
            evolved = sum(1 for c in children[iid] if children.get(c))
            ratio = evolved / n_kids
            if ratio <= th.flat_hub_evolve_ratio:
                _add("flat_hub", "high" if n_kids >= 4 * th.flat_hub_fanout else "medium",
                     f"{n_kids} sibling ideas but only {evolved} "
                     f"({ratio * 100:.0f}%) were evolved further — the search is "
                     f"broad but shallow here.",
                     f"Evolve the best children of {iid} into multi-step lineages "
                     f"instead of only spawning more siblings.", node_id=iid)

        # evolution-contract flags (aggregated per parent to stay bounded)
        if n_kids >= 1:
            zero = [c for c in children[iid] if evo[c]["delta_size"] == 0]
            if len(zero) >= th.eval_child_min:
                _add("pseudo_evolution", "medium",
                     f"{len(zero)} of {n_kids} children share {iid}'s exact config "
                     f"— these are re-runs/evals, not evolutions of it.",
                     f"Re-label these as evaluations, or change a parameter so each "
                     f"child is a real variation of {iid}.", node_id=iid)
            unjust = [c for c in children[iid]
                      if evo[c]["delta_size"] > 0 and not evo[c]["rationale"]]
            if len(unjust) >= th.unjustified_min:
                _add("unjustified_branch", "low",
                     f"{len(unjust)} children of {iid} changed the config but recorded "
                     f"no rationale for the change.",
                     f"Record a rationale (the hypothesis behind the change) on each "
                     f"variation of {iid}.", node_id=iid)

        # saturated parent: many children, none beat the parent
        if n_kids >= th.saturated_fanout:
            improving = sum(1 for c in children[iid] if (_delta(c) or 0) > 0)
            if improving == 0 and goodness[iid] is not None:
                _add("over_researched", "medium",
                     f"{n_kids} child ideas, none improved on the parent.",
                     f"Stop expanding {iid}; pivot to a different approach.", node_id=iid)

        # stagnant lineage: long non-improving refinement chain ending here
        if (stagnant_run.get(iid, 0) >= th.over_stagnant_run
                and depth[iid] >= th.over_min_chain and n_kids == 0):
            _add("over_researched", "medium",
                 f"{stagnant_run[iid]} consecutive refinements (depth {depth[iid]}) "
                 f"with no improvement.",
                 f"Abandon this lineage; best gains were upstream of {iid}.",
                 node_id=iid)

    # failed clusters: maximal subtrees that are mostly failures
    for iid in in_tree:
        size = subtree_size.get(iid, 1)
        if size < th.fail_cluster_min_size:
            continue
        frac = failed_count.get(iid, 0) / size
        if frac < th.fail_cluster_frac:
            continue
        p = parent_of.get(iid)
        # only flag the top of a failed region to avoid nested duplicates
        if p is not None and subtree_size.get(p, 1) and \
                failed_count.get(p, 0) / subtree_size.get(p, 1) >= th.fail_cluster_frac:
            continue
        _add("failed_cluster", "high" if frac >= 0.85 else "medium",
             f"{int(round(frac * 100))}% of {size} ideas in this subtree failed.",
             f"Investigate a systemic error under {iid} before spending more compute.",
             node_id=iid)

    # missing coverage: under-explored taxonomy buckets (region-level)
    coverage: Dict[str, Any] = {}
    for dim in ("category", "approach_family"):
        counts: Dict[str, int] = defaultdict(int)
        for iid in in_tree:
            counts[(by_id[iid].get(dim) or "unknown")] += 1
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])
        # Keep payload bounded; summarise the long tail of free-text buckets.
        coverage[dim] = dict(ordered[:th.coverage_max_buckets])
        if len(ordered) > th.coverage_max_buckets:
            coverage[dim]["…(other)"] = sum(c for _, c in ordered[th.coverage_max_buckets:])
        # Only flag thin buckets for a *controlled* taxonomy. A dimension with
        # too many distinct values is unnormalised free text — flagging it is noise.
        if 2 <= len(counts) <= th.coverage_max_buckets:
            mean = sum(counts.values()) / len(counts)
            thin = th.coverage_thin_ratio * mean
            flagged = 0
            for bucket, c in sorted(counts.items(), key=lambda kv: kv[1]):
                if c >= thin or flagged >= th.coverage_max_flags:
                    break
                _add("missing_coverage", "low",
                     f"'{bucket}' {dim} has only {c} idea(s) "
                     f"(mean {mean:.0f} per {dim}).",
                     f"Generate more ideas in {dim}='{bucket}'.",
                     region=f"{dim}={bucket}")
                flagged += 1

    # rank problems for the attention list
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    problems.sort(key=lambda q: (sev_rank.get(q["severity"], 3), q["kind"]))

    # ---- node cap (keep largest trees) ----
    truncated = False
    if len(nodes) > th.max_nodes:
        keep_roots: set = set()
        kept = 0
        for r in roots:
            keep_roots.add(r)
            kept += subtree_size.get(r, 1)
            if kept >= th.max_nodes:
                break
        keep_ids = set()
        for r in keep_roots:
            stack = [r]
            while stack:
                x = stack.pop()
                keep_ids.add(x)
                stack.extend(children.get(x, ()))
        nodes = [n for n in nodes if n["id"] in keep_ids]
        problems = [q for q in problems
                    if q["node_id"] is None or q["node_id"] in keep_ids]
        truncated = True
        node_index = {n["id"]: n for n in nodes}

    edges = [{"source": n["parent"], "target": n["id"]}
             for n in nodes if n["parent"] and n["parent"] in node_index]

    all_depths = [n["depth"] for n in nodes]
    status_counts: Dict[str, int] = defaultdict(int)
    for r in by_id.values():
        status_counts[(r.get("status") or "unknown").lower()] += 1

    # refinement success rate over scored parent/child pairs
    pc_pairs = improved_pairs = 0
    for c, p in parent_of.items():
        if goodness.get(c) is not None and goodness.get(p) is not None:
            pc_pairs += 1
            if goodness[c] > goodness[p]:  # type: ignore[operator]
                improved_pairs += 1

    # evolution rate: share of parented ideas that were themselves evolved further
    # (i.e. are an intermediate node, not just a leaf hanging off a hub). Low = the
    # search is breadth-first sibling spawning rather than multi-step evolution.
    intermediate = sum(1 for c in parent_of if children.get(c))
    evolution_rate = round(intermediate / len(parent_of), 4) if parent_of else None

    # evolution-contract accounting over rendered edges
    rendered_ids = {n["id"] for n in nodes}
    evo_types: Dict[str, int] = defaultdict(int)
    n_edges = n_judged = n_contract_ok = n_zero_delta = n_no_rationale = n_undiffable = 0
    for iid in rendered_ids:
        e = evo.get(iid)
        if not e:
            continue
        evo_types[e["evolution_type"]] += 1
        if parent_of.get(iid) is None:
            continue
        n_edges += 1
        if e["contract_ok"] is None:
            n_undiffable += 1            # config missing: delta unjudgeable
            continue
        n_judged += 1
        if e["contract_ok"]:
            n_contract_ok += 1
        if "zero_delta" in e["contract_violations"]:
            n_zero_delta += 1
        if "no_rationale" in e["contract_violations"]:
            n_no_rationale += 1
    # Rate is over edges we could actually judge (configs present on both sides).
    genuine_evolution_rate = round(n_contract_ok / n_judged, 4) if n_judged else None

    # depth-yield curve: does searching *deeper* actually produce scored results?
    # Buckets depths >= 10 into a single "10+" row to keep the payload small.
    dy: Dict[Any, dict] = {}
    for n in nodes:
        b = n["depth"] if n["depth"] < 10 else 10
        slot = dy.setdefault(b, {"depth": b, "n": 0, "scored": 0, "best_metric": None})
        slot["n"] += 1
        if n["metric"] is not None:
            slot["scored"] += 1
            bm = slot["best_metric"]
            g = goodness.get(n["id"])
            if g is not None and (bm is None or g > _goodness(bm, lower_is_better)):
                slot["best_metric"] = n["metric"]
    depth_yield = []
    for b in sorted(dy):
        row = dy[b]
        row["label"] = f"{b}+" if b == 10 else str(b)
        row["scored_frac"] = round(row["scored"] / row["n"], 4) if row["n"] else 0.0
        depth_yield.append(row)

    n_leaves = sum(1 for n in nodes if n["n_children"] == 0)
    fanout = [n["n_children"] for n in nodes if n["n_children"] > 0]
    research_efficiency = compute_research_efficiency(
        n_total=len(by_id),
        n_scored=n_scored,
        status_counts={k: v for k, v in status_counts.items()},
        fanout=fanout,
        n_leaves=n_leaves,
        n_intermediate=intermediate,
        refinement_success_rate=(round(improved_pairs / pc_pairs, 4) if pc_pairs else None),
        evolution_rate=evolution_rate,
        depth_yield=depth_yield,
        th=th,
    )

    return {
        "metric": {"name": metric_name, "lower_is_better": lower_is_better},
        "nodes": nodes,
        "edges": edges,
        "problems": problems,
        "coverage": coverage,
        "stats": {
            "n_total": len(by_id),
            "n_in_tree": len(in_tree),
            "n_rendered": len(nodes),
            "n_scored": n_scored,
            "n_roots": len(roots),
            "max_depth": max(all_depths) if all_depths else 0,
            "mean_depth": round(sum(all_depths) / len(all_depths), 3) if all_depths else 0,
            "status_counts": dict(status_counts),
            "refinement_success_rate":
                round(improved_pairs / pc_pairs, 4) if pc_pairs else None,
            "refinement_pairs": pc_pairs,
            "evolution_rate": evolution_rate,
            "intermediate_nodes": intermediate,
            "n_edges": n_edges,
            "judged_edges": n_judged,
            "undiffable_edges": n_undiffable,
            "genuine_evolution_rate": genuine_evolution_rate,
            "contract_ok_edges": n_contract_ok,
            "zero_delta_edges": n_zero_delta,
            "no_rationale_edges": n_no_rationale,
            "evolution_types": dict(evo_types),
            "truncated": truncated,
            "problem_counts": _count_by(problems, "kind"),
        },
        "research_efficiency": research_efficiency,
        "thresholds": asdict(th),
    }


def _subtree_size(iid: str, children: Dict[str, List[str]]) -> int:
    size, stack = 0, [iid]
    while stack:
        x = stack.pop()
        size += 1
        stack.extend(children.get(x, ()))
    return size


def _count_by(items: List[dict], key: str) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for it in items:
        out[it[key]] += 1
    return dict(out)


# ---------------------------------------------------------------------------
#  Data source: idea_lake.db -> rows + a config-driven metric resolver
# ---------------------------------------------------------------------------

def load_rows_from_lake(db_path: str) -> List[dict]:
    """Read genealogy-relevant columns from an idea_lake.db (read-only)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ideas)")}
    want = ["idea_id", "title", "parent", "category", "approach_family",
            "status", "priority", "eval_metrics", "training_time", "created_at",
            "config", "hypothesis", "kind"]
    sel = ", ".join(c for c in want if c in cols)
    rows: List[dict] = []
    for row in conn.execute(f"SELECT {sel} FROM ideas"):
        d = dict(row)
        em = d.get("eval_metrics")
        if isinstance(em, str) and em:
            try:
                d["eval_metrics"] = json.loads(em)
            except (json.JSONDecodeError, TypeError):
                d["eval_metrics"] = {}
        elif not isinstance(em, dict):
            d["eval_metrics"] = {}
        # The rationale (why this idea evolves its parent) is stored as the
        # idea's hypothesis; expose it under the generic name the builder uses.
        d["rationale"] = d.get("hypothesis")
        # Parse the stored config (YAML) so edges can be diffed structurally.
        cfg_raw = d.get("config")
        if isinstance(cfg_raw, str) and cfg_raw.strip():
            try:
                import yaml
                parsed = yaml.safe_load(cfg_raw)
                d["config"] = parsed if isinstance(parsed, dict) else None
            except Exception:
                d["config"] = None
        elif not isinstance(cfg_raw, dict):
            d["config"] = None
        rows.append(d)
    conn.close()
    return rows


def make_metric_resolver(cfg: Optional[dict]):
    """Return (metric_of, lower_is_better, metric_name) from project report config.

    Generic: the metric KEY is derived from the configured primary metric (or the
    source dotpath of its report column), and looked up inside each idea's
    harvested ``eval_metrics``. Direction comes from ``report.sort``.
    """
    report = (cfg or {}).get("report", {}) or {}
    primary = report.get("primary_metric", "score")
    lower_is_better = str(report.get("sort", "descending")).lower().startswith("asc")

    # Candidate keys to probe inside eval_metrics, most specific first.
    keys: List[str] = [primary]
    for col in report.get("columns", []) or []:
        if col.get("key") == primary:
            src = col.get("source", "")
            if ":" in src:
                keys.append(src.split(":", 1)[1])  # e.g. "avg_wer"
    keys += ["avg_wer", "wer", "score", "test_accuracy"]  # last-resort fallbacks

    def metric_of(row: dict) -> Optional[float]:
        em = row.get("eval_metrics") or {}
        if not isinstance(em, dict):
            return None
        for k in keys:
            v = em.get(k)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                return float(v)
        return None

    return metric_of, lower_is_better, primary


def build_from_lake(db_path: str, cfg: Optional[dict] = None,
                    thresholds: Optional[Thresholds] = None) -> Dict[str, Any]:
    rows = load_rows_from_lake(db_path)
    metric_of, lib, name = make_metric_resolver(cfg)
    return build_search_path(
        rows, metric_of=metric_of, lower_is_better=lib, metric_name=name,
        thresholds=thresholds or Thresholds.from_config(cfg),
    )


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build research search-path graph JSON.")
    ap.add_argument("--db", default=".orze/idea_lake.db")
    ap.add_argument("--config", default=None, help="orze.yaml for metric/thresholds")
    ap.add_argument("--json", default=None, help="output path (default: stdout)")
    args = ap.parse_args()

    cfg = None
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not load config {args.config}: {e}")

    data = build_from_lake(args.db, cfg)
    out = json.dumps(data, indent=2, default=str)
    if args.json:
        with open(args.json, "w") as f:
            f.write(out)
        s = data["stats"]
        print(f"wrote {args.json}: {s['n_rendered']} nodes, "
              f"{len(data['problems'])} problems, {s['n_roots']} roots")
    else:
        print(out)


if __name__ == "__main__":
    _main()
