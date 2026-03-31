"""Orze Smart Suggestions — rule-based idea generation from experiment analysis.

Part of orze (free, open source). Converts retrospection insights into
concrete experiment ideas without requiring an LLM or API key.

    orze free:  Smart Suggestions (rule-based, deterministic)
    orze-pro:   Research Agents (LLM-driven, creative, hypothesis-based)

CALLING SPEC:
    suggest_ideas(analysis, best_config, cfg) -> list[dict]
        Takes experiment_analysis output + best config, returns list of
        idea dicts ready to write to ideas.md.

    write_suggestions(ideas_path, suggestions) -> int
        Appends suggestion ideas to ideas.md. Returns count written.

Smart Suggestions generates ideas from:
    1. Regression fixes — if dataset X regressed, try lower LoRA scale
    2. Tradeoff exploration — if A and B anti-correlate, sweep the tradeoff
    3. Scale refinement — fine-grained sweep around optimal scale
    4. Best-config variations — systematic perturbations of winning config
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger("orze")

_BRAND = "Orze Smart Suggestions"


def suggest_ideas(analysis: dict, best_config: dict,
                  cfg: dict) -> List[dict]:
    """Generate experiment ideas from analysis insights.

    Returns list of dicts with keys: id, title, priority, config, reason.
    """
    if not analysis:
        return []

    ideas = []
    best = analysis.get("best", {})
    baseline = analysis.get("baseline", {})
    regressions = analysis.get("regressions", [])
    improvements = analysis.get("improvements", [])
    patterns = analysis.get("patterns", [])

    # Strategy 1: Fix regressions by reducing LoRA scale
    for reg in regressions:
        if reg["delta"] > 0.1:
            current_scale = best_config.get("lora_scale", 1.0)
            lora_path = best_config.get("lora_path", "")
            if lora_path and current_scale > 0.1:
                # Try a lower scale to reduce regression
                new_scale = round(max(0.1, current_scale - 0.1), 2)
                idea_cfg = dict(best_config)
                idea_cfg["lora_scale"] = new_scale
                ideas.append({
                    "title": f"Fix {reg['dataset']} regression: scale {current_scale}->{new_scale}",
                    "priority": "high",
                    "family": "optimization",
                    "config": idea_cfg,
                    "reason": (f"{_BRAND}: {reg['dataset']} regressed "
                               f"+{reg['delta']:.2f}% vs baseline. "
                               f"Lower LoRA scale reduces domain mismatch."),
                })

    # Strategy 2: Tradeoff sweep — if two datasets anti-correlate,
    # generate a fine-grained scale sweep between current and baseline
    tradeoffs = [p for p in patterns if "TRADEOFF" in p]
    if tradeoffs:
        current_scale = best_config.get("lora_scale", 1.0)
        lora_path = best_config.get("lora_path", "")
        if lora_path:
            for step in [-0.05, -0.15, 0.05]:
                new_scale = round(current_scale + step, 3)
                if 0.0 < new_scale <= 1.5 and new_scale != current_scale:
                    idea_cfg = dict(best_config)
                    idea_cfg["lora_scale"] = new_scale
                    ideas.append({
                        "title": f"Tradeoff sweep: scale={new_scale}",
                        "priority": "medium",
                        "family": "optimization",
                        "config": idea_cfg,
                        "reason": (f"{_BRAND}: Tradeoff detected between datasets. "
                                   f"Fine-grained scale sweep to find optimal balance."),
                    })

    # Strategy 3: If no regressions and improvements exist, push harder
    if not regressions and improvements:
        current_scale = best_config.get("lora_scale", 1.0)
        lora_path = best_config.get("lora_path", "")
        if lora_path and current_scale <= 1.2:
            new_scale = round(current_scale + 0.05, 3)
            idea_cfg = dict(best_config)
            idea_cfg["lora_scale"] = new_scale
            ideas.append({
                "title": f"Push further: scale {current_scale}->{new_scale} (no regressions)",
                "priority": "medium",
                "family": "optimization",
                "config": idea_cfg,
                "reason": (f"{_BRAND}: All datasets improved or held. "
                           f"Safe to increase LoRA influence."),
            })

    # Strategy 4: If scale patterns show monotonic effect, extrapolate
    for p in patterns:
        if "higher LoRA scale = better" in p:
            current_scale = best_config.get("lora_scale", 1.0)
            lora_path = best_config.get("lora_path", "")
            if lora_path and current_scale < 1.3:
                for bump in [0.05, 0.10]:
                    new_scale = round(current_scale + bump, 3)
                    idea_cfg = dict(best_config)
                    idea_cfg["lora_scale"] = new_scale
                    ideas.append({
                        "title": f"Scale extrapolation: {new_scale} (monotonic improvement)",
                        "priority": "medium",
                        "family": "optimization",
                        "config": idea_cfg,
                        "reason": (f"{_BRAND}: Analysis shows higher scale = better. "
                                   f"Extrapolating trend."),
                    })

    # Deduplicate by config hash
    seen = set()
    unique = []
    for idea in ideas:
        cfg_str = json.dumps(idea["config"], sort_keys=True)
        h = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
        if h not in seen:
            seen.add(h)
            idea["id"] = f"idea-ss{h}"
            unique.append(idea)
    return unique


def write_suggestions(ideas_path: Path, suggestions: List[dict]) -> int:
    """Append Smart Suggestion ideas to ideas.md."""
    if not suggestions:
        return 0

    lines = []
    for s in suggestions:
        clean_cfg = {k: v for k, v in s["config"].items()
                     if not k.startswith("_")}
        lines.append(f"\n## {s['id']}: {s['title']}")
        lines.append(f"- **Priority**: {s['priority']}")
        lines.append(f"- **Approach Family**: {s.get('family', 'optimization')}")
        lines.append(f"- **Generated by**: {_BRAND}")
        lines.append(f"- **Reason**: {s['reason']}")
        lines.append(f"```yaml")
        lines.append(yaml.dump(clean_cfg, default_flow_style=False).rstrip())
        lines.append(f"```")
        lines.append("")

    with open(ideas_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("[%s] Generated %d ideas from experiment analysis",
                _BRAND, len(suggestions))
    return len(suggestions)
