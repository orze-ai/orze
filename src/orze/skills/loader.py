"""Skill loader — composable prompt fragments for Orze roles.

CALLING SPEC:
    compose_skills(role_cfg, project_root, template_vars=None) -> str
        role_cfg:       dict with a 'skills' list
        project_root:   Path to project directory
        template_vars:  dict of {key: value} for substitution, or None to skip
        returns:        composed prompt string

    compose_skills_with_manifest(..., context_for_skill=None, strict=False)
        Return text and the actual included source descriptors from one read.
        The optional callback supplies per-source trigger context; no activation
        state is written here. Strict native gating is explicitly opt-in.

    load_builtin(name) -> Skill
        name: one of 'core', 'research', 'ops', 'setup'
        returns: Skill namedtuple(name, content)
        raises: FileNotFoundError if name is unknown

SKILLS GRAMMAR:
    role_cfg['skills'] is a list of references in composition order.
    Each ref is one of:
      - '@core' | '@research' | '@ops' | '@setup' | '@release'
            built-in .skill.md shipped with orze.
      - '@sop:<name>'
            static SOP bundled in orze-pro (orze_pro/sops/<name>.skill.md).
            Resolved via optional import of orze_pro.skills.bundled; if
            orze-pro is not installed the ref is skipped with a warning.
      - './path.md' | 'path.md'
            project-local file relative to project_root. Used for dynamic
            (project-authored) SOPs or free-form prompt fragments.

    Skills declaring frontmatter 'order' are sorted ascending; skills
    declaring 'trigger' are gated by the role's _trigger_context. The
    composed output substitutes template_vars if provided.
"""

import logging
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger("orze")

Skill = namedtuple("Skill", ["name", "content"])

_SKILLS_DIR = Path(__file__).parent

# Valid built-in skill names
_BUILTINS = {"core", "research", "ops", "setup", "release"}


def parse_frontmatter(text: str) -> tuple:
    """Split ---yaml--- header from content.

    Returns (meta_dict, body_str). If no frontmatter, returns ({}, text).
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()


def load_builtin(name: str) -> Skill:
    """Load a built-in skill from this package's .skill.md files."""
    if name not in _BUILTINS:
        raise FileNotFoundError(
            f"Unknown built-in skill '@{name}'. "
            f"Available: {', '.join(sorted('@' + b for b in _BUILTINS))}")
    path = _SKILLS_DIR / f"{name}.skill.md"
    if not path.exists():
        raise FileNotFoundError(f"Built-in skill file not found: {path}")
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    return Skill(name=meta.get("name", name), content=body)


def load_file(path: Path) -> Skill:
    """Load a skill from a project file.

    Plain .md without frontmatter gets name from filename stem.
    """
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    name = meta.get("name", path.stem)
    return Skill(name=name, content=body)


def _substitute(text: str, template_vars: Dict[str, str]) -> str:
    """Safe template var substitution using str.replace."""
    for k, v in template_vars.items():
        text = text.replace(f"{{{k}}}", str(v))
    return text


def compose_skills(role_cfg: dict, project_root: Path,
                   template_vars: Optional[Dict[str, str]] = None) -> str:
    """Legacy string API: preserve default warning/include gate semantics."""
    return compose_skills_with_manifest(role_cfg, project_root, template_vars)["text"]


def compose_skills_with_manifest(role_cfg: dict, project_root: Path,
                                 template_vars=None, *, context_for_skill=None,
                                 strict=False) -> dict:
    """Compose once; describe only actual nonempty included source fragments.

    Source hashes are provenance, not periodic activation keys. The caller owns
    any successful-launch ACK. Strict mode excludes unknown/invalid gates and
    unusable callback context; legacy manual/plateau_or_new/zero gates remain
    explicitly diagnosed always-included compatibility aliases.
    """
    from orze.skills.composition import compose_with_manifest
    return compose_with_manifest(
        role_cfg, project_root, template_vars,
        context_for_skill=context_for_skill, strict=strict)
