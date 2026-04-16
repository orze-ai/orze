"""Skill registry — discover SOP skills with frontmatter metadata.

Each project skill file lives at ``<project_root>/skills/<name>.skill.md``
with YAML frontmatter declaring SOP metadata:

    ---
    id: sop-da-anomaly              # required for registry tracking
    name: anomaly_driven_hypotheses # display name
    role: data_analyst              # which role composes this skill
    order: 30                       # composition order (low first)
    produces: [results/_X.md, ...]  # files this skill writes
    consumed_by: [research, thinker]# downstream roles
    requires: [sop-da-base, ...]    # upstream skill ids that must have produced
    trigger: periodic_research_cycles(5)  # activation gate; None/"always" = always
    overrides: sop-other-id         # replace another skill (conflict check)
    ---

CALLING SPEC:
    discover_skills(project_root) -> List[SkillMetadata]
        Scan <project_root>/skills/*.skill.md, parse frontmatter, return
        metadata objects for skills that declare an ``id``.

    validate_wiring(skills) -> List[WiringIssue]
        Run cross-skill checks:
          - every ``requires`` points to a known skill
          - every ``produces`` has a consumer (via consumed_by or requires)
          - every ``overrides`` target exists
        Returns list of issues (severity='error' or 'warning').
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from orze.skills.loader import parse_frontmatter


@dataclass
class SkillMetadata:
    id: str
    name: str
    role: Optional[str] = None
    order: int = 100
    produces: List[str] = field(default_factory=list)
    consumed_by: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    trigger: Optional[str] = None
    overrides: Optional[str] = None
    path: Optional[Path] = None


@dataclass
class WiringIssue:
    severity: str
    skill_id: str
    message: str


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def discover_skills(project_root: Path) -> List[SkillMetadata]:
    skills_dir = project_root / "skills"
    if not skills_dir.exists():
        return []
    results: List[SkillMetadata] = []
    for path in sorted(skills_dir.glob("*.skill.md")):
        try:
            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        sid = meta.get("id")
        if not sid:
            continue  # not a registry-tracked SOP
        results.append(SkillMetadata(
            id=str(sid),
            name=str(meta.get("name", path.stem)),
            role=meta.get("role"),
            order=int(meta.get("order", 100)),
            produces=_as_list(meta.get("produces")),
            consumed_by=_as_list(meta.get("consumed_by")),
            requires=_as_list(meta.get("requires")),
            trigger=meta.get("trigger"),
            overrides=meta.get("overrides"),
            path=path,
        ))
    return results


def validate_wiring(skills: List[SkillMetadata]) -> List[WiringIssue]:
    by_id = {s.id: s for s in skills}
    issues: List[WiringIssue] = []

    # Check 1: requires must point to known skills
    for s in skills:
        for req in s.requires:
            if req not in by_id:
                issues.append(WiringIssue(
                    severity="error",
                    skill_id=s.id,
                    message=(f"{s.id} requires '{req}' but no such skill "
                             f"is registered"),
                ))

    # Check 2: produces must have a consumer. A consumer is any other
    # skill whose ``requires`` or ``consumed_by`` references this skill id.
    # (consumed_by values may be role names too — treat role-name matches
    # as consumers as well: if any other skill has role == x and this
    # skill's consumed_by contains x, that counts.)
    referenced_ids: Set[str] = set()
    referenced_roles: Set[str] = set()
    for s in skills:
        for r in s.requires:
            referenced_ids.add(r)
    all_roles = {s.role for s in skills if s.role}

    for s in skills:
        if not s.produces:
            continue
        has_consumer = False
        # case A: another skill requires this skill id
        if s.id in referenced_ids:
            has_consumer = True
        # case B: consumed_by lists a role that exists in the registry
        if not has_consumer:
            for target in s.consumed_by:
                if target in all_roles or target in by_id:
                    has_consumer = True
                    break
        if not has_consumer:
            issues.append(WiringIssue(
                severity="warning",
                skill_id=s.id,
                message=(f"{s.id} produces {s.produces} but has no consumer "
                         f"(no skill requires it and consumed_by targets "
                         f"{s.consumed_by or 'are empty'})"),
            ))

    # Check 3: overrides target must exist
    for s in skills:
        if s.overrides and s.overrides not in by_id:
            issues.append(WiringIssue(
                severity="error",
                skill_id=s.id,
                message=(f"{s.id} overrides '{s.overrides}' but target "
                         f"does not exist"),
            ))

    return issues
