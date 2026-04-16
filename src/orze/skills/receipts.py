"""Execution receipts for role/skill runs.

A Receipt records, per role invocation:
- role name, cycle number, start/end timestamps
- which skills were declared in the composed prompt
- which skills showed evidence of execution (declared outputs changed)
- the declared produced-file list per skill, used to compute evidence

CALLING SPEC:
    Receipt(role, cycle, started_at, ended_at, skills_declared,
            skills_evidenced, outputs)
        Dataclass. to_json() / from_json() for persistence.

    write_receipt(receipt, path) -> None
        Write a receipt as JSON to path (creates parent dirs).

    read_receipt(path) -> Receipt
        Load a receipt from JSON.

    snapshot_mtimes(declared_outputs, project_root) -> Dict[str, float]
        Capture mtimes of every declared output path. Missing files map to 0.0.
        Call BEFORE the role runs.

    compute_evidenced_skills(declared_outputs, mtime_snapshot_before,
                             project_root) -> List[str]
        Return skill ids whose declared outputs changed after the role ran.
        Call AFTER the role runs, passing the pre-run snapshot.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Receipt:
    role: str
    cycle: int
    started_at: float
    ended_at: float
    skills_declared: List[str]
    skills_evidenced: List[str]
    outputs: Dict[str, List[str]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Receipt":
        data = json.loads(text)
        return cls(**data)


def write_receipt(receipt: Receipt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(receipt.to_json(), encoding="utf-8")


def read_receipt(path: Path) -> Receipt:
    return Receipt.from_json(path.read_text(encoding="utf-8"))


def _abspath(rel: str, project_root: Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else project_root / p


def snapshot_mtimes(declared_outputs: Dict[str, List[str]],
                    project_root: Path) -> Dict[str, float]:
    """Capture mtimes for all declared outputs before a role runs."""
    snap: Dict[str, float] = {}
    for paths in declared_outputs.values():
        for rel in paths:
            abspath = _abspath(rel, project_root)
            try:
                snap[str(abspath)] = abspath.stat().st_mtime
            except FileNotFoundError:
                snap[str(abspath)] = 0.0
    return snap


def compute_evidenced_skills(declared_outputs: Dict[str, List[str]],
                             mtime_snapshot_before: Dict[str, float],
                             project_root: Path) -> List[str]:
    """Return skills whose declared outputs changed after the role ran."""
    evidenced: List[str] = []
    for skill_id, paths in declared_outputs.items():
        for rel in paths:
            abspath = _abspath(rel, project_root)
            key = str(abspath)
            mtime_before = mtime_snapshot_before.get(key, 0.0)
            try:
                mtime_after = abspath.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime_after > mtime_before:
                evidenced.append(skill_id)
                break
    return evidenced
