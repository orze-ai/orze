"""Detect train script changes and auto-generate smoke-test ideas.

CALLING SPEC:
    from orze.engine.code_change import CodeChangeDetector

    detector = CodeChangeDetector(train_script, results_dir, ideas_path, base_config)
    detector.check()  # call every N iterations; appends ideas if train.py changed

When train.py changes (git diff or mtime):
  1. Extract new/modified config keys from the diff (cfg.get("key") patterns)
  2. Generate smoke-test ideas using those keys
  3. Append to ideas.md with priority=critical
  4. Log what was detected and generated

This ensures that manual fixes and code evolution changes get automatically tested
without waiting for the research agent to discover them.
"""

import hashlib
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import yaml

logger = logging.getLogger("orze")

# Match cfg.get("key"), cfg["key"], config.get("key"), config["key"]
_CFG_KEY_RE = re.compile(
    r'''(?:cfg|config|c)\s*(?:\.get\s*\(\s*|[\[]\s*)["']([a-z_][a-z0-9_]*)["']''',
    re.IGNORECASE,
)


def _extract_config_keys_from_diff(diff_text: str) -> Set[str]:
    """Extract config key names from added lines in a git diff."""
    keys = set()
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            keys.update(_CFG_KEY_RE.findall(line))
    # Filter out obvious non-config keys
    ignore = {"steps", "batch_size", "lr", "seed", "max_samples", "status",
              "idea_id", "results_dir", "config", "key", "value", "default"}
    return keys - ignore


def _get_train_script_diff(train_script: Path, last_hash: str) -> Tuple[str, str]:
    """Get git diff of train script since last check. Returns (diff, new_hash)."""
    try:
        content = train_script.read_text()
        new_hash = hashlib.md5(content.encode()).hexdigest()
        if new_hash == last_hash:
            return "", new_hash

        # Try git diff
        result = subprocess.run(
            ["git", "diff", "HEAD~1", "--", str(train_script)],
            capture_output=True, text=True, timeout=5,
            cwd=str(train_script.parent),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout, new_hash

        # Fallback: if no git diff, return the full file as "added"
        # (this handles uncommitted changes)
        result2 = subprocess.run(
            ["git", "diff", "--", str(train_script)],
            capture_output=True, text=True, timeout=5,
            cwd=str(train_script.parent),
        )
        if result2.returncode == 0 and result2.stdout.strip():
            return result2.stdout, new_hash

        return "", new_hash
    except Exception:
        try:
            content = train_script.read_text()
            return "", hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return "", last_hash


def _generate_smoke_ideas(new_keys: Set[str], base_config: Path,
                          ideas_path: Path) -> List[str]:
    """Generate smoke-test ideas that exercise new config keys."""
    if not new_keys or not base_config.exists():
        return []

    try:
        base = yaml.safe_load(base_config.read_text()) or {}
    except Exception:
        return []

    ideas = []
    idea_num = int(time.time()) % 100000

    # Generate one idea per new key with a simple true/1.0 value
    for key in sorted(new_keys):
        if key in base:
            continue  # Already in base config — not a new feature

        idea_num += 1
        idea_id = f"idea-smoke{idea_num}"

        # Build config: base + new key enabled
        cfg = dict(base)
        cfg[key] = True  # Most new features are boolean flags
        cfg["steps"] = min(base.get("steps", 200000), 50000)  # Short run
        cfg["seed"] = 42

        config_yaml = yaml.dump(cfg, default_flow_style=False)
        md = (
            f"## {idea_id}: Smoke test for new config key `{key}`\n"
            f"- **Priority**: critical\n"
            f"- **Category**: smoke_test\n"
            f"- **Hypothesis**: Auto-generated to test new/fixed code path `{key}`\n"
            f"```yaml\n{config_yaml}```\n"
        )
        ideas.append(md)

    return ideas


class CodeChangeDetector:
    """Watches train script for changes and generates smoke-test ideas."""

    def __init__(self, train_script: str, results_dir: Path,
                 ideas_path: Path, base_config: str):
        self._script = Path(train_script)
        self._results_dir = results_dir
        self._ideas_path = Path(ideas_path)
        self._base_config = Path(base_config)
        self._last_hash = ""
        self._tested_keys: Set[str] = set()  # Keys we've already generated tests for

        # Initialize hash
        if self._script.exists():
            try:
                self._last_hash = hashlib.md5(
                    self._script.read_text().encode()
                ).hexdigest()
            except Exception:
                pass

    def check(self) -> int:
        """Check for train script changes. Returns number of ideas generated."""
        if not self._script.exists():
            return 0

        diff, new_hash = _get_train_script_diff(self._script, self._last_hash)
        if not diff:
            self._last_hash = new_hash
            return 0

        self._last_hash = new_hash

        # Extract new config keys from the diff
        new_keys = _extract_config_keys_from_diff(diff)
        # Only test keys we haven't already generated tests for
        untested = new_keys - self._tested_keys
        if not untested:
            logger.info("[CODE_CHANGE] train.py changed but no new config keys found")
            return 0

        logger.info("[CODE_CHANGE] train.py changed — new config keys: %s",
                     sorted(untested))

        # Generate and append smoke-test ideas
        ideas_md = _generate_smoke_ideas(
            untested, self._base_config, self._ideas_path)
        if not ideas_md:
            return 0

        try:
            from orze.core.fs import _fs_lock, _fs_unlock
            lock_dir = self._results_dir / ".ideas_md.lock"
            locked = _fs_lock(lock_dir, stale_seconds=60)
            try:
                with open(self._ideas_path, "a", encoding="utf-8") as f:
                    f.write("\n")
                    for md in ideas_md:
                        f.write(md)
                        f.write("\n")
            finally:
                if locked:
                    _fs_unlock(lock_dir)
        except Exception as e:
            logger.warning("[CODE_CHANGE] Failed to append ideas: %s", e)
            return 0

        self._tested_keys.update(untested)
        logger.info("[CODE_CHANGE] Generated %d smoke-test ideas for: %s",
                     len(ideas_md), sorted(untested))
        return len(ideas_md)
