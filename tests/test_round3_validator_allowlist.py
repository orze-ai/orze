"""Round-3 (cosmetic harden): the config validator's allowlist must
cover every top-level key we ship in the `orze setup` template.

Background
==========
The validator in ``orze.core.config.validate_config`` warns about any
top-level YAML key that is neither in ``DEFAULT_CONFIG`` nor in the
local ``_KNOWN_EXTRAS`` allowlist. The intent is "catch typos like
``poll_interval`` when you meant ``poll``."

Bug
---
``orze setup`` writes a starter ``orze.yaml`` from a Python f-string
template in ``orze.cli_setup``. That template included three
top-level keys (``nested_config_whitelist``, ``metric_harvest``,
``sweep_allowlist``) that were missing from the validator allowlist
— so every fresh install produced cosmetic ``Unknown config key``
warnings on keys orze itself had just emitted. Worst case the user
would (correctly!) read the warning and delete the key, regressing
features (e.g. ``nested_config_whitelist`` is the gate that lets
phases.py preserve user-curated nested config sections during
LLM-driven config edits).

This test prevents regression by parsing the setup template and
asserting every top-level key it emits is recognised by the
validator.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from orze.core.config import DEFAULT_CONFIG, _validate_config as validate_config
from orze import cli_setup as _cli_setup


def _render_setup_template() -> str:
    """Pull the orze.yaml f-string template(s) out of cli_setup.py
    and concatenate them. There are currently two — one for projects
    with orze-pro enabled, one without — and we want to validate
    both. Strategy: find every yaml_content f-string triple-quoted
    block in the cli_setup source.
    """
    src = Path(_cli_setup.__file__).read_text()
    matches = re.findall(
        r'yaml_content\s*=\s*f"""\\?\n(.*?)"""',
        src,
        re.DOTALL,
    )
    assert matches, "could not locate any yaml_content templates in cli_setup.py"
    raw = "\n".join(matches)
    # The template uses {train_script} etc.; substitute placeholders
    # with sensible stand-ins so YAML stays parseable. Doubled braces
    # ({{ }}) survive .format unchanged → real { } in output.
    placeholders = {
        "train_script": "train.py",
        "results_dir": "orze_results",
        "ideas_file": "ideas.md",
        "goal_file": "GOAL.md",
        "base_config": "configs/base.yaml",
        "python": "/usr/bin/python3",
    }
    # Best-effort fill; unknown placeholders become empty strings.
    class _SafeDict(dict):
        def __missing__(self, key):  # type: ignore[override]
            return ""
    return raw.format_map(_SafeDict(placeholders))


def test_setup_template_keys_are_all_recognised():
    rendered = _render_setup_template()
    # Tolerate the template having stray un-substituted placeholders
    # — we only need the top-level structure.
    try:
        cfg = yaml.safe_load(rendered) or {}
    except yaml.YAMLError:
        # Fallback: extract top-level keys via regex.
        cfg = {
            m.group(1): None
            for m in re.finditer(r"^([a-z_][a-z0-9_]*)\s*:", rendered, re.MULTILINE)
        }
    assert isinstance(cfg, dict)
    template_keys = set(cfg.keys())

    # validate_config returns (errors, warnings); we look for
    # any "Unknown config key '<k>'" warning whose <k> appears in
    # the template. Those are the false-positives we want to ban.
    errors, warnings = validate_config(cfg)
    bad = []
    for w in warnings:
        m = re.search(r"Unknown config key '([^']+)'", w)
        if m and m.group(1) in template_keys:
            bad.append(m.group(1))
    assert not bad, (
        f"validator warns on keys shipped by `orze setup` template: {sorted(bad)}. "
        f"Add them to _KNOWN_EXTRAS in orze/core/config.py."
    )


def test_known_extras_covers_template_specials():
    """Spot-check the three keys that motivated this round."""
    pass  # imports already at module level
    # Re-run validate_config against a minimal dict containing each
    # key; warnings should not flag them as unknown.
    sample = {
        "nested_config_whitelist": ["model"],
        "metric_harvest": {"columns": []},
        "sweep_allowlist": ["train.py"],
    }
    _errors, warnings = validate_config(sample)
    for k in sample:
        for w in warnings:
            assert f"Unknown config key '{k}'" not in w, (
                f"key {k} should be in validator allowlist"
            )
