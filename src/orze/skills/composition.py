"""Single-read skill composition and actual-inclusion provenance.

CALLING SPEC:
    compose_with_manifest(role_cfg, project_root, template_vars=None, *,
                          context_for_skill=None, strict=False) -> dict
        Implementation of loader.compose_skills_with_manifest. Returns text,
        included source descriptors and excluded descriptors/reasons. Does not
        persist activation state, reread sources for ACK, or inspect experiments.

The source hash binds loaded text, not an attempt or execution receipt. Package
resource keys survive installation-root changes; local keys use canonical paths.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
import re

import yaml

from orze.skills import loader


def _load(ref, project_root):
    if ref.startswith("@sop:"):
        from orze_pro.skills.bundled import load_bundled_skill
        name = ref[len("@sop:"):]
        text, path = load_bundled_skill(name)
        key = f"orze_pro:sops/{name}.skill.md"
        builtin = False
    elif ref.startswith("@"):
        name = ref[1:]
        if name not in loader._BUILTINS:
            raise FileNotFoundError(f"Unknown built-in skill {ref!r}")
        path = loader._SKILLS_DIR / f"{name}.skill.md"
        text = path.read_text(encoding="utf-8")
        key = f"orze:skills/{name}.skill.md"
        builtin = True
    else:
        path = Path(ref)
        if not path.is_absolute():
            path = Path(project_root) / path
        text = path.read_text(encoding="utf-8")
        key = "file:" + str(path.resolve())
        builtin = False
    descriptor = {
        "source_key": key, "requested_ref": ref,
        "resolved_source": str(Path(path).resolve()), "trigger": None,
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return text, descriptor, builtin


def _metadata(text, strict):
    if not strict:
        return loader.parse_frontmatter(text)
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("skill_frontmatter_unterminated")
    meta = yaml.safe_load(parts[1])
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError("skill_frontmatter_not_mapping")
    return meta, parts[2].strip()


def _gate_kind(trigger):
    if trigger is None:
        return "always"
    if not isinstance(trigger, str):
        return "invalid"
    text = trigger.strip()
    if not text or text.lower() == "always":
        return "always"
    if text in ("manual", "plateau_or_new"):
        return "legacy_always"
    if re.fullmatch(r"(?:periodic_research_cycles|on_plateau)\(\s*0+\s*\)", text):
        return "legacy_always"
    if (re.fullmatch(r"(?:periodic_research_cycles|on_plateau)\(\s*\d+\s*\)", text)
            or re.fullmatch(r"on_file\(\s*(.+?)\s*\)", text)):
        return "conditional"
    return "invalid"


def _context(descriptor, callback, common, strict):
    context = callback(deepcopy(descriptor)) if callback is not None else common
    if strict and not isinstance(context, Mapping):
        raise ValueError("skill_context_not_mapping")
    if strict and _gate_kind(descriptor["trigger"]) == "conditional":
        trigger = descriptor["trigger"].strip()
        fields = (("research_cycles", "last_activation_cycle")
                  if trigger.startswith("periodic_research_cycles(") else
                  ("plateau_patience",) if trigger.startswith("on_plateau(") else ())
        for field in fields:
            value = context.get(field, 0)
            if type(value) is not int or value < 0:
                raise ValueError("skill_context_count_invalid")
    return context


def compose_with_manifest(role_cfg, project_root, template_vars=None, *,
                          context_for_skill=None, strict=False):
    result = {"text": "", "included": [], "excluded": []}
    if "skills" not in role_cfg:
        return result
    references = role_cfg["skills"]
    if not isinstance(references, list):
        loader.logger.warning("'skills' must be a list, got %s", type(references).__name__)
        return result
    try:
        from orze.skills.triggers import evaluate_trigger
    except ImportError:
        evaluate_trigger = None
    common = role_cfg.get("_trigger_context", {})
    if not strict:
        common = common or {}
    loaded = []
    for reference in references:
        ref = str(reference).strip()
        try:
            text, descriptor, builtin = _load(ref, project_root)
        except (ImportError, OSError) as exc:
            loader.logger.warning("Skill %s unavailable (%s)", ref, type(exc).__name__)
            continue

        def exclude(reason):
            result["excluded"].append({**deepcopy(descriptor), "reason": reason})
            loader.logger.warning("Skill %s excluded: %s", ref, reason)

        try:
            meta, body = _metadata(text, strict)
        except (yaml.YAMLError, ValueError) as exc:
            if not strict:
                raise
            exclude("skill_frontmatter_invalid")
            continue
        descriptor["trigger"] = deepcopy(meta.get("trigger")) if meta else None
        # The legacy builtin branch ignored its frontmatter order/trigger.
        # Native strict composition uses the actual declared metadata.
        effective_meta = {} if builtin and not strict else meta
        trigger = effective_meta.get("trigger") if effective_meta else None
        rendered = loader._substitute(body, template_vars) if template_vars else body
        if strict and not rendered.strip():
            exclude("skill_content_empty")
            continue
        kind = _gate_kind(trigger)
        if strict and kind == "invalid":
            exclude("skill_trigger_invalid")
            continue
        try:
            context = _context(descriptor, context_for_skill, common, strict)
        except Exception:
            if not strict:
                raise
            exclude("skill_context_invalid")
            continue
        if strict and kind == "legacy_always":
            loader.logger.warning(
                "Skill %s trigger %r is legacy always-included compatibility, "
                "not a manual or evidence gate", ref, trigger)
            enabled = True
        elif evaluate_trigger is None:
            if strict and trigger is not None:
                exclude("skill_trigger_evaluator_unavailable")
                continue
            enabled = True
        elif trigger is None:
            enabled = True
        else:
            try:
                enabled = evaluate_trigger(trigger, context)
            except (ValueError, TypeError, AttributeError, OverflowError) as exc:
                if strict:
                    exclude("skill_trigger_context_invalid")
                    continue
                # Preserve the original ValueError warn/include behavior.
                # Bad other types retain the old failure semantics below.
                if not isinstance(exc, ValueError):
                    raise
                loader.logger.warning("Skill %s: unknown trigger %r — including", ref, trigger)
                enabled = True
        if not enabled:
            exclude("skill_trigger_not_met")
            continue
        try:
            order = int((effective_meta or {}).get("order", 100))
        except (TypeError, ValueError, OverflowError):
            if not strict:
                raise
            exclude("skill_order_invalid")
            continue
        loaded.append((order, body, rendered, descriptor))
    loaded.sort(key=lambda row: row[0])
    # Legacy empty fragments/separators and whole-string substitution stay
    # byte-compatible. They never masquerade as actual included skill content.
    if strict:
        result["text"] = "\n\n---\n\n".join(row[2] for row in loaded)
    else:
        text = "\n\n---\n\n".join(row[1] for row in loaded)
        result["text"] = loader._substitute(text, template_vars) if template_vars else text
    result["included"] = [deepcopy(row[3]) for row in loaded if row[2].strip()]
    loader.logger.info("Composed %d nonempty skill fragments", len(result["included"]))
    return result
