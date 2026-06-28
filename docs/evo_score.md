# Evo Score — orze's research-efficiency metric

**Status:** implemented (admin panel) · **Scope:** task-agnostic engine metric

## What it is

**Evo Score** is orze's single top-level measure of *research efficiency* — "is the
search producing results per unit of research effort, and is it compounding?" It is
deliberately **not** compute efficiency (GPU/VRAM utilization) and **not** the
project's task metric (e.g. WER). It scores how well orze *searches*, on a
**0–100 scale with an A–F grade**, and is surfaced as the first KPI on the
Overview tab plus a full panel in the Research Tree tab.

It is computed live by `compute_research_efficiency()` in
`orze/reporting/search_path.py` and served at `GET /api/research_efficiency`
(also embedded in `/api/search_path`).

## How it is computed

Evo Score is a weighted blend of five components, each normalized to `0..1`
(higher = better), combined into a 0–100 score:

| Component | Meaning | Weight |
|---|---|---|
| **Yield** | scored results ÷ ideas produced (saturates at `eff_yield_target`, default 10%) | 1.0 |
| **Refine success** | refinements that actually beat their parent | 1.5 |
| **Depth utilization** | parented ideas that are themselves evolved further (search compounds vs sprays flat) | 1.5 |
| **Diversity** | `1 − share of edges held by the single biggest hub` | 1.0 |
| **Reliability** | `1 − failure rate` | 1.0 |

```
Evo Score = 100 × Σ(componentᵢ.score × weightᵢ) / Σ(weightᵢ)     # Σweight = 6
```

All weights, the yield target, and grade cutoffs are configurable under
`report.search_path` (`eff_w_*`, `eff_yield_target`, …).

### Grade cutoffs

| Grade | Score |
|---|---|
| A | ≥ 85 |
| B | ≥ 70 |
| C | ≥ 55 |
| D | ≥ 40 |
| F | < 40 |

The panel additionally reports the **explore-vs-exploit** split, **hub
concentration** (top-1 / top-5 share, Gini), and a **depth-yield curve** (do
deeper lineages actually produce scored results?).

## Current state of the live run — 16 / F

| Component | Value | Normalized score |
|---|---|---|
| Yield | 1.2% scored | 0.12 |
| Refine success | 0% improve | 0.00 |
| Depth utilization | 5.3% evolved | 0.05 |
| Diversity | hub holds 56% of edges | 0.44 |
| Reliability | 71% failure rate | 0.29 |
| **Evo Score** | | **15.6 → grade F** |

Supporting signals: single champion hub holds **56%** of all edges (max fan-out
2182, Gini 0.90); scored results appear only at depth 1–2; exploit share 5.3%
(3688 explore leaves vs 206 exploit/intermediate nodes). The metric quantifies
what the tree shows visually: a **flat star** — a bandit hammering one arm, with
no compounding.

## What Grade A looks like

Grade A (≥85) requires every component at roughly **0.85+**. Concretely:

| Component | F today | Grade-A target |
|---|---|---|
| Yield | 1.2% | **≥ 10%** of ideas produce a scored result |
| Refine success | 0% | **≥ 80%** of refinements improve on their parent |
| Depth utilization | 5.3% | **≥ 80%** — winners spawn deeper refinements |
| Diversity | 0.44 (hub 56%) | **≥ 0.85** — no single hub > ~15% of edges |
| Reliability | 0.29 (71% fail) | **≥ 0.90** — ≤ 10% of runs fail/error |

**Picture of an A-grade run:** not today's flat star, but a **broad, deep tree** —
many independent lineages, each winner promoted and refined several levels deep,
≥4 in 5 refinements actually improving, ≥1 in 10 ideas producing a scored
result, and ≤1 in 10 runs erroring.

Worked example that lands at A:

```
yield 1.0, success 0.8, depth 0.8, diversity 0.85, reliability 0.9
→ 100 × (1·1.0 + 1.5·0.8 + 1.5·0.8 + 1·0.85 + 1·0.9) / 6 = 85.8  → A
```

(All components at 0.9 → exactly 90.)

## The biggest lever

**Depth + refine success** carry weight 1.5 each — together they dominate the
score. The single most effective change is to make **winners spawn deeper
refinements that genuinely beat their parent**, instead of producing thousands
of shallow sibling variations off one champion hub. Capping hub fan-out
(diversity) and reducing run failures (reliability) are the next levers.

## Proposed next steps

1. **Wire Evo Score into the engine loop** — log it to `status.json` /
   admin_cache each iteration so it is trended over time, not only computed on
   demand.
2. **Let the policy act on it** — when Evo Score (or its depth/diversity
   components) drops, throttle the champion hub's fan-out and promote winning
   depth-1 variations into deeper lineages so search compounds.
3. **Make `config` storage mandatory at idea creation** so undiffable edges
   shrink and the evolution contract's `genuine_evolution_rate` becomes
   meaningful (today ~2235 edges are undiffable due to empty `config`).
4. **Tune weights / thresholds** (`report.search_path`) once the loop reacts to
   the metric, to match the desired explore/exploit balance.
