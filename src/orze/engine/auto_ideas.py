"""Auto-generate parameter variation ideas when the queue is empty.

CALLING SPEC:
    generate_variations(results_dir, cfg, lake, max_ideas=5) -> int
        Reads the top completed experiment, generates +-perturbations
        of its numeric config values, and appends them to ideas.md.
        Returns number of ideas generated.

    This is the FREE research loop — no LLM needed, just systematic
    exploration around the current best. orze-pro replaces this with
    intelligent, hypothesis-driven research agents.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger("orze")


def _get_best_config(results_dir: Path, cfg: dict) -> Optional[dict]:
    """Find the best completed experiment and return its config."""
    report_cfg = cfg.get("report", {})
    primary = report_cfg.get("primary_metric", "avg_wer")
    sort_asc = report_cfg.get("sort", "descending") == "ascending"

    best_val = float("inf") if sort_asc else float("-inf")
    best_cfg = None

    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mf = d / "metrics.json"
        cf = d / "idea_config.yaml"
        if not mf.exists() or not cf.exists():
            continue
        try:
            m = json.loads(mf.read_text())
            if m.get("status") != "COMPLETED":
                continue
            val = m.get(primary)
            if val is None or not isinstance(val, (int, float)):
                continue
            better = val < best_val if sort_asc else val > best_val
            if better:
                best_val = val
                best_cfg = yaml.safe_load(cf.read_text()) or {}
                best_cfg["_idea_id"] = d.name
                best_cfg["_primary_val"] = val
        except Exception:
            continue

    return best_cfg


# Keys that are safe to perturb (numeric hyperparameters)
_PERTURBABLE = {
    "lora_scale", "lora_rank", "lora_alpha",
    "learning_rate", "lr", "weight_decay",
    "max_new_tokens", "temperature", "top_p",
    "epochs", "num_train_steps", "grad_accum",
    "batch_size", "max_samples",
}

# Keys to never perturb
_SKIP = {
    "model_path", "whisper_path", "boson_multimodal_path",
    "eval_data_dir", "lora_path", "train_data",
    "user_prompt", "system_prompt", "text_normalizer",
    "_idea_id", "_primary_val",
}


def _perturbations(key: str, value) -> List:
    """Generate perturbation values for a config key."""
    if not isinstance(value, (int, float)):
        return []
    if key in _SKIP:
        return []
    if value == 0:
        return []

    # Boolean-like (0/1)
    if isinstance(value, bool):
        return [not value]

    # Scale factors (0-2 range)
    if "scale" in key:
        candidates = [round(value - 0.05, 3), round(value + 0.05, 3)]
        return [c for c in candidates if 0.0 < c <= 2.0 and c != value]

    # Learning rate (log scale)
    if key in ("learning_rate", "lr"):
        return [value * 0.5, value * 2.0]

    # Integer params
    if isinstance(value, int):
        if value >= 4:
            return [max(1, value // 2), value * 2]
        return []

    # Generic float
    return [round(value * 0.8, 6), round(value * 1.2, 6)]


def generate_variations(results_dir: Path, cfg: dict,
                        lake=None, max_ideas: int = 5) -> int:
    """Generate parameter variation ideas from the best experiment."""
    best = _get_best_config(results_dir, cfg)
    if best is None:
        return 0

    parent_id = best.pop("_idea_id", "?")
    parent_val = best.pop("_primary_val", "?")

    # Find perturbable keys
    variations = []
    for key, value in best.items():
        if key in _SKIP:
            continue
        if key not in _PERTURBABLE and not isinstance(value, (int, float)):
            continue
        for new_val in _perturbations(key, value):
            var = dict(best)
            var[key] = new_val
            variations.append((key, value, new_val, var))

    if not variations:
        logger.debug("No perturbable parameters found in best config")
        return 0

    # Deduplicate against existing ideas in the lake
    if lake:
        existing = lake.get_all_ids()
    else:
        existing = set()

    # Generate idea markdown
    ideas_path = Path(cfg.get("ideas_file", "ideas.md"))
    generated = 0
    lines = []

    for key, old_val, new_val, var_cfg in variations[:max_ideas]:
        # Create a unique idea ID
        import hashlib
        cfg_hash = hashlib.md5(
            json.dumps(var_cfg, sort_keys=True).encode()
        ).hexdigest()[:6]
        idea_id = f"idea-v{cfg_hash}"

        if idea_id in existing:
            continue

        direction = "+" if new_val > old_val else "-"
        title = f"Variation: {key}={new_val} ({direction} from {old_val}, parent: {parent_id})"

        # Remove non-config keys
        clean_cfg = {k: v for k, v in var_cfg.items()
                     if not k.startswith("_")}

        lines.append(f"\n## {idea_id}: {title}")
        lines.append(f"- **Priority**: low")
        lines.append(f"- **Approach Family**: optimization")
        lines.append(f"- **Parent**: {parent_id}")
        lines.append(f"```yaml")
        lines.append(yaml.dump(clean_cfg, default_flow_style=False).rstrip())
        lines.append(f"```")
        lines.append("")
        generated += 1

    if lines:
        with open(ideas_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Auto-generated %d parameter variations from %s (%.4f)",
                    generated, parent_id, parent_val)

    return generated
