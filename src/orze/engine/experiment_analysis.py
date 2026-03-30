"""Cross-experiment regression analysis for retrospection.

CALLING SPEC:
    analyze_experiments(results_dir, cfg, top_n=10) -> dict
        Compares the top-N experiments per metric column, detects
        regressions and improvements relative to the baseline (worst
        top-N entry or explicit baseline), and returns structured
        insights for the research agent.

    format_insights(analysis) -> str
        Formats the analysis dict into a human-readable string
        suitable for appending to research agent prompts or
        retrospection output.

The analysis dict contains:
    {
        "best": {"id": str, "avg": float, "per_dataset": dict},
        "baseline": {"id": str, "avg": float, "per_dataset": dict},
        "regressions": [{"dataset": str, "best_val": float,
                         "baseline_val": float, "delta": float,
                         "worst_offender": str, "pattern": str}],
        "improvements": [...same structure...],
        "patterns": [str],  # cross-experiment strategic insights
        "suggested_actions": [str],
    }
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("orze")


def _load_completed_experiments(results_dir: Path, cfg: dict,
                                 limit: int = 200) -> List[dict]:
    """Load recent completed experiments with per-dataset metrics."""
    report_cfg = cfg.get("report", {})
    primary = report_cfg.get("primary_metric", "avg_wer")
    sort_asc = report_cfg.get("sort", "descending") == "ascending"

    experiments = []
    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            if m.get("status") != "COMPLETED":
                continue
            pval = m.get(primary)
            if pval is None or not isinstance(pval, (int, float)):
                continue

            # Extract per-dataset metrics
            per_ds = {}
            for k, v in m.items():
                if k.startswith("wer_") and isinstance(v, (int, float)):
                    per_ds[k] = v

            # Extract config info
            cfg_path = d / "idea_config.yaml"
            idea_cfg = {}
            if cfg_path.exists():
                try:
                    import yaml
                    idea_cfg = yaml.safe_load(cfg_path.read_text()) or {}
                except Exception:
                    pass

            experiments.append({
                "id": d.name,
                "primary": pval,
                "per_dataset": per_ds,
                "mtime": mf.stat().st_mtime,
                "config": idea_cfg,
                "lora_path": idea_cfg.get("lora_path", ""),
                "lora_scale": idea_cfg.get("lora_scale", 1.0),
            })
        except Exception:
            continue

    # Sort by primary metric
    experiments.sort(key=lambda x: x["primary"],
                     reverse=not sort_asc)
    return experiments[:limit]


def analyze_experiments(results_dir: Path, cfg: dict,
                        top_n: int = 10) -> Optional[dict]:
    """Run cross-experiment analysis. Returns None if insufficient data."""
    experiments = _load_completed_experiments(results_dir, cfg, limit=200)
    if len(experiments) < 3:
        return None

    report_cfg = cfg.get("report", {})
    columns = report_cfg.get("columns", [])
    ds_columns = [c for c in columns if c.get("key", "").startswith("wer_")]

    if not ds_columns:
        return None

    # Best experiment (top of sorted list)
    best = experiments[0]

    # Find baseline: experiment with no lora_path (if any), else worst of top-N
    top = experiments[:top_n]
    baseline = None
    for exp in experiments:
        if not exp.get("lora_path"):
            baseline = exp
            break
    if baseline is None and len(top) > 1:
        baseline = top[-1]
    if baseline is None:
        return None

    # Per-dataset analysis
    regressions = []
    improvements = []

    for col in ds_columns:
        key = col["key"]
        label = col.get("label", key.replace("wer_", ""))

        best_val = best["per_dataset"].get(key)
        base_val = baseline["per_dataset"].get(key)

        if best_val is None or base_val is None:
            continue

        delta = best_val - base_val

        # Find which experiment is worst on this dataset
        worst_on_ds = max(top, key=lambda x: x["per_dataset"].get(key, 0))
        best_on_ds = min(top, key=lambda x: x["per_dataset"].get(key, float("inf")))

        entry = {
            "dataset": label,
            "key": key,
            "best_val": best_val,
            "baseline_val": base_val,
            "delta": round(delta, 2),
            "best_experiment": best_on_ds["id"],
            "best_experiment_val": best_on_ds["per_dataset"].get(key),
            "worst_experiment": worst_on_ds["id"],
            "worst_experiment_val": worst_on_ds["per_dataset"].get(key),
        }

        if delta > 0.1:  # regression > 0.1%
            # Find pattern: do experiments with more training data regress?
            entry["pattern"] = _find_regression_pattern(
                key, top, baseline)
            regressions.append(entry)
        elif delta < -0.1:  # improvement > 0.1%
            improvements.append(entry)

    # Cross-experiment patterns
    patterns = _find_strategic_patterns(top, ds_columns, baseline)

    # Generate suggested actions
    actions = _suggest_actions(regressions, improvements, patterns, best)

    return {
        "best": {
            "id": best["id"],
            "primary": best["primary"],
            "per_dataset": best["per_dataset"],
            "config_summary": {
                "lora_path": best.get("lora_path", ""),
                "lora_scale": best.get("lora_scale", 1.0),
            },
        },
        "baseline": {
            "id": baseline["id"],
            "primary": baseline["primary"],
            "per_dataset": baseline["per_dataset"],
        },
        "regressions": regressions,
        "improvements": improvements,
        "patterns": patterns,
        "suggested_actions": actions,
        "experiment_count": len(experiments),
    }


def _find_regression_pattern(key: str, experiments: list,
                              baseline: dict) -> str:
    """Try to explain WHY a dataset regressed across experiments."""
    base_val = baseline["per_dataset"].get(key, 0)
    if base_val == 0:
        return "no baseline"

    # Group by lora_path
    by_lora = defaultdict(list)
    for exp in experiments:
        lora = exp.get("lora_path", "none") or "none"
        # Shorten path to checkpoint name
        if "/" in lora:
            lora = lora.rstrip("/").rsplit("/", 2)[-2] if "/best" in lora else lora.rsplit("/", 1)[-1]
        val = exp["per_dataset"].get(key)
        if val is not None:
            by_lora[lora].append(val)

    # Check if all LoRA variants regress on this dataset
    regressed_loras = []
    for lora, vals in by_lora.items():
        if lora == "none":
            continue
        avg = sum(vals) / len(vals)
        if avg > base_val + 0.1:
            regressed_loras.append((lora, avg - base_val))

    if regressed_loras:
        worst = max(regressed_loras, key=lambda x: x[1])
        return (f"All LoRA variants regress on this dataset. "
                f"Worst: {worst[0]} (+{worst[1]:.2f}%). "
                f"Likely caused by domain mismatch in training data.")
    return "mixed — some configs help, others hurt"


def _find_strategic_patterns(experiments: list, ds_columns: list,
                              baseline: dict) -> List[str]:
    """Find high-level strategic patterns across experiments."""
    patterns = []

    if len(experiments) < 3:
        return patterns

    # Check if more training data = more regression on untrained datasets
    by_scale = defaultdict(list)
    for exp in experiments:
        scale = exp.get("lora_scale", 1.0)
        by_scale[scale].append(exp)

    if len(by_scale) > 1:
        # Compare avg regression across scales
        for col in ds_columns:
            key = col["key"]
            base_val = baseline["per_dataset"].get(key, 0)
            if base_val == 0:
                continue
            scale_impact = []
            for scale, exps in sorted(by_scale.items()):
                vals = [e["per_dataset"].get(key, 0) for e in exps if key in e["per_dataset"]]
                if vals:
                    avg_delta = sum(vals) / len(vals) - base_val
                    scale_impact.append((scale, avg_delta))
            if len(scale_impact) > 1:
                # Check monotonic relationship
                deltas = [d for _, d in scale_impact]
                if all(deltas[i] <= deltas[i+1] for i in range(len(deltas)-1)):
                    ds_name = col.get("label", key)
                    patterns.append(
                        f"{ds_name}: higher LoRA scale = worse. "
                        f"Scale {scale_impact[0][0]}→{scale_impact[-1][0]}: "
                        f"delta {scale_impact[0][1]:+.2f}→{scale_impact[-1][1]:+.2f}%")
                elif all(deltas[i] >= deltas[i+1] for i in range(len(deltas)-1)):
                    ds_name = col.get("label", key)
                    patterns.append(
                        f"{ds_name}: higher LoRA scale = better. "
                        f"Scale {scale_impact[0][0]}→{scale_impact[-1][0]}: "
                        f"delta {scale_impact[0][1]:+.2f}→{scale_impact[-1][1]:+.2f}%")

    # Check tradeoff: datasets that anti-correlate
    ds_keys = [c["key"] for c in ds_columns]
    for i, k1 in enumerate(ds_keys):
        for k2 in ds_keys[i+1:]:
            vals1 = [e["per_dataset"].get(k1) for e in experiments if k1 in e["per_dataset"] and k2 in e["per_dataset"]]
            vals2 = [e["per_dataset"].get(k2) for e in experiments if k1 in e["per_dataset"] and k2 in e["per_dataset"]]
            if len(vals1) >= 3:
                # Simple correlation check
                mean1 = sum(vals1) / len(vals1)
                mean2 = sum(vals2) / len(vals2)
                cov = sum((a - mean1) * (b - mean2) for a, b in zip(vals1, vals2)) / len(vals1)
                std1 = (sum((a - mean1)**2 for a in vals1) / len(vals1)) ** 0.5
                std2 = (sum((b - mean2)**2 for b in vals2) / len(vals2)) ** 0.5
                if std1 > 0 and std2 > 0:
                    corr = cov / (std1 * std2)
                    if corr < -0.7:
                        n1 = next((c.get("label", k1) for c in ds_columns if c["key"] == k1), k1)
                        n2 = next((c.get("label", k2) for c in ds_columns if c["key"] == k2), k2)
                        patterns.append(
                            f"TRADEOFF: {n1} and {n2} are anti-correlated "
                            f"(r={corr:.2f}). Improving one hurts the other.")

    return patterns


def _suggest_actions(regressions: list, improvements: list,
                     patterns: list, best: dict) -> List[str]:
    """Generate concrete action suggestions based on analysis."""
    actions = []

    if regressions:
        # Prioritize by delta (biggest regression first)
        regressions.sort(key=lambda x: x["delta"], reverse=True)
        worst = regressions[0]
        actions.append(
            f"PRIORITY: Fix {worst['dataset']} regression "
            f"(+{worst['delta']:.2f}% vs baseline). "
            f"Consider: adding {worst['dataset']} training data, "
            f"or reducing LoRA scale to limit regression.")

    tradeoffs = [p for p in patterns if "TRADEOFF" in p]
    if tradeoffs:
        actions.append(
            "TRADEOFF detected — parameter sweeps won't help. "
            "Try: per-dataset inference strategies, post-processing, "
            "or training data that covers BOTH sides of the tradeoff.")

    scale_patterns = [p for p in patterns if "LoRA scale" in p]
    if scale_patterns:
        actions.append(
            "LoRA scale has monotonic effect on some datasets. "
            "Optimal scale is a compromise — try fine-grained sweep "
            "around current best (±0.05 steps).")

    if not regressions and improvements:
        actions.append(
            "No regressions detected! Current approach is working. "
            "Consider: scaling up training data, longer training, "
            "or higher LoRA rank.")

    return actions


def format_insights(analysis: dict) -> str:
    """Format analysis into human-readable text for research agents."""
    if not analysis:
        return "Insufficient data for cross-experiment analysis."

    lines = [
        "=== CROSS-EXPERIMENT ANALYSIS ===",
        f"Analyzed {analysis['experiment_count']} experiments.",
        f"Best: {analysis['best']['id']} (avg={analysis['best']['primary']:.2f}%)",
        f"Baseline: {analysis['baseline']['id']} (avg={analysis['baseline']['primary']:.2f}%)",
        "",
    ]

    if analysis["improvements"]:
        lines.append("IMPROVEMENTS vs baseline:")
        for imp in analysis["improvements"]:
            lines.append(
                f"  {imp['dataset']}: {imp['baseline_val']:.2f}% → "
                f"{imp['best_val']:.2f}% ({imp['delta']:+.2f}%)"
                f"  [best: {imp['best_experiment']}={imp['best_experiment_val']:.2f}%]")
        lines.append("")

    if analysis["regressions"]:
        lines.append("REGRESSIONS vs baseline:")
        for reg in analysis["regressions"]:
            lines.append(
                f"  {reg['dataset']}: {reg['baseline_val']:.2f}% → "
                f"{reg['best_val']:.2f}% ({reg['delta']:+.2f}%)"
                f"  [pattern: {reg.get('pattern', '?')}]")
        lines.append("")

    if analysis["patterns"]:
        lines.append("STRATEGIC PATTERNS:")
        for p in analysis["patterns"]:
            lines.append(f"  • {p}")
        lines.append("")

    if analysis["suggested_actions"]:
        lines.append("SUGGESTED ACTIONS:")
        for a in analysis["suggested_actions"]:
            lines.append(f"  → {a}")

    return "\n".join(lines)
