"""Sealed file integrity verification and metric validation.

CALLING SPEC:
    compute_sealed_hashes(file_list: list[str]) -> dict[str, str]
        SHA-256 hash each file in file_list.
        Returns {path: hex_digest}. Logs warning for missing files.

    write_sealed_manifest(results_dir: Path, hashes: dict[str, str]) -> None
        Write .sealed_hashes JSON manifest to results_dir.

    load_sealed_manifest(results_dir: Path) -> dict[str, str]
        Read .sealed_hashes manifest. Returns {} if not found.

    verify_sealed_files(file_list: list[str], manifest: dict[str, str]) -> list[str]
        Re-hash each file and compare to manifest.
        Returns list of changed/missing file paths. Empty = all OK.

    validate_metrics(metrics: dict, cfg: dict) -> tuple[bool, str]
        Check metrics.json values for NaN, inf, or out-of-range.
        Uses cfg["report"]["sort"] to determine metric direction.
        Returns (is_valid, reason). reason is "" if valid.
"""

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("orze")

_MANIFEST_FILE = ".sealed_hashes"


def compute_sealed_hashes(file_list: List[str]) -> Dict[str, str]:
    """Compute SHA-256 hashes for each file in the list."""
    hashes = {}
    for fpath in file_list:
        p = Path(fpath)
        if not p.exists():
            logger.warning("Sealed file does not exist: %s", fpath)
            continue
        try:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            hashes[fpath] = h
        except OSError as e:
            logger.warning("Could not hash sealed file %s: %s", fpath, e)
    return hashes


def write_sealed_manifest(results_dir: Path,
                          hashes: Dict[str, str]) -> None:
    """Write the sealed hashes manifest to results_dir."""
    manifest_path = results_dir / _MANIFEST_FILE
    try:
        manifest_path.write_text(
            json.dumps(hashes, indent=2), encoding="utf-8")
        logger.info("Sealed manifest written: %d files tracked", len(hashes))
    except OSError as e:
        logger.error("Could not write sealed manifest: %s", e)


def load_sealed_manifest(results_dir: Path) -> Dict[str, str]:
    """Load the sealed hashes manifest. Returns {} if not found."""
    manifest_path = results_dir / _MANIFEST_FILE
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def verify_sealed_files(file_list: List[str],
                        manifest: Dict[str, str]) -> List[str]:
    """Verify sealed files haven't changed since manifest was written.

    Returns list of file paths that have changed or are missing.
    Empty list means all files are intact.
    """
    if not manifest:
        return []

    changed = []
    for fpath in file_list:
        expected = manifest.get(fpath)
        if expected is None:
            continue  # file wasn't in manifest (maybe added after startup)

        p = Path(fpath)
        if not p.exists():
            changed.append(fpath)
            continue

        try:
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != expected:
                changed.append(fpath)
        except OSError:
            changed.append(fpath)

    return changed


def validate_metrics(metrics: dict, cfg: dict) -> Tuple[bool, str]:
    """Validate metric values in a metrics.json dict.

    Checks for NaN, inf, and optionally validates ranges.

    Args:
        metrics: Parsed metrics.json dict.
        cfg: orze config dict (uses cfg["report"] and cfg["metric_validation"]).

    Returns:
        (is_valid, reason) — reason is "" if valid.
    """
    if not metrics or metrics.get("status") != "COMPLETED":
        return True, ""  # only validate completed metrics

    validation_cfg = cfg.get("metric_validation", {})
    reject_nan = validation_cfg.get("reject_nan", True)
    reject_inf = validation_cfg.get("reject_inf", True)
    min_values = validation_cfg.get("min_value", {})
    max_values = validation_cfg.get("max_value", {})

    for key, val in metrics.items():
        if key in ("status", "timestamp", "error", "idea_id"):
            continue
        if not isinstance(val, (int, float)):
            continue

        if reject_nan and math.isnan(val):
            return False, f"Metric '{key}' is NaN"
        if reject_inf and math.isinf(val):
            return False, f"Metric '{key}' is inf"

        if key in min_values and val < min_values[key]:
            return False, f"Metric '{key}'={val} below minimum {min_values[key]}"
        if key in max_values and val > max_values[key]:
            return False, f"Metric '{key}'={val} above maximum {max_values[key]}"

    return True, ""
