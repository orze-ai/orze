"""Skill-level activation gates.

Triggers do NOT schedule role runs (that is ``cycle_interval`` on the role
in ``orze.yaml``). A trigger decides whether a given skill is INCLUDED in
the composed prompt for a given cycle. A skill whose trigger evaluates
False is silently omitted; the role still runs with its remaining skills.

Supported grammar:
    None | "always"                      -> always active
    "periodic_research_cycles(N)"        -> active once per N research cycles
    "on_file(path)"                      -> active when path exists
    "on_plateau(N)"                      -> active when plateau_patience >= N

CALLING SPEC:
    evaluate_trigger(expr, context) -> bool
        expr: None or one of the grammar forms above.
        context: dict with keys used by the trigger type, e.g.
                 - research_cycles, last_activation_cycle
                 - plateau_patience
        raises ValueError on unknown trigger expressions.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional


def evaluate_trigger(trigger: Optional[str],
                     context: Dict[str, Any]) -> bool:
    if trigger is None:
        return True
    t = str(trigger).strip()
    if not t or t.lower() == "always":
        return True

    m = re.fullmatch(r"periodic_research_cycles\(\s*(\d+)\s*\)", t)
    if m:
        interval = int(m.group(1))
        rc = int(context.get("research_cycles", 0))
        last = int(context.get("last_activation_cycle", 0))
        return rc - last >= interval

    m = re.fullmatch(r"on_file\(\s*(.+?)\s*\)", t)
    if m:
        return Path(m.group(1).strip()).exists()

    m = re.fullmatch(r"on_plateau\(\s*(\d+)\s*\)", t)
    if m:
        threshold = int(m.group(1))
        return int(context.get("plateau_patience", 0)) >= threshold

    raise ValueError(f"Unknown trigger expression: {trigger!r}")
