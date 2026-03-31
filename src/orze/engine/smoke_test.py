"""Startup smoke test — verifies train script contract before burning GPU hours.

CALLING SPEC:
    from orze.engine.smoke_test import run_smoke_test
    passed, errors = run_smoke_test(cfg, results_dir)

Runs a single-sample test idea through the full pipeline:
  1. Writes idea_config.yaml with a sentinel key
  2. Launches train.py with max_samples=1
  3. Verifies: config received, metrics.json written, primary metric present
  4. Checks for anomalies: unknown config keys silently dropped, wrong avg
  5. Cleans up test artifacts

Returns (True, []) on success, (False, [error_strings]) on failure.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import yaml
from pathlib import Path

logger = logging.getLogger("orze")

_SMOKE_IDEA_ID = "_smoke_test"
_SENTINEL_KEY = "_smoke_sentinel"
_SENTINEL_VALUE = "orze_contract_check_42"
_TIMEOUT = 300  # 5 min max for a single sample


def run_smoke_test(cfg: dict, results_dir: Path) -> tuple:
    """Run end-to-end smoke test. Returns (passed: bool, errors: list[str])."""
    errors = []
    idea_dir = results_dir / _SMOKE_IDEA_ID

    # Clean up any previous smoke test
    if idea_dir.exists():
        shutil.rmtree(idea_dir, ignore_errors=True)

    try:
        idea_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write idea_config.yaml with sentinel + minimal eval
        base_cfg_path = cfg.get("base_config", "configs/base.yaml")
        base_cfg = {}
        if Path(base_cfg_path).exists():
            base_cfg = yaml.safe_load(Path(base_cfg_path).read_text()) or {}

        # Override to run minimal eval
        idea_cfg = {
            _SENTINEL_KEY: _SENTINEL_VALUE,
            "max_samples": 1,
        }
        # Keep only first eval task for speed
        if base_cfg.get("eval_tasks"):
            idea_cfg["eval_tasks"] = [base_cfg["eval_tasks"][0]]

        idea_cfg_path = idea_dir / "idea_config.yaml"
        idea_cfg_path.write_text(yaml.dump(idea_cfg, default_flow_style=False))

        # 2. Launch train.py
        python = cfg.get("python", sys.executable)
        train_script = cfg["train_script"]
        cmd = [
            python, train_script,
            "--idea-id", _SMOKE_IDEA_ID,
            "--results-dir", str(results_dir),
            "--ideas-md", cfg.get("ideas_file", "ideas.md"),
            "--config", base_cfg_path,
        ]

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only for smoke test

        logger.info("[SMOKE TEST] Running: %s", " ".join(cmd[-4:]))
        t0 = time.time()

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT, env=env,
        )
        elapsed = time.time() - t0

        # 3. Check exit code
        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else ""
            stdout_tail = result.stdout[-500:] if result.stdout else ""
            errors.append(
                f"Train script exited with code {result.returncode}.\n"
                f"stderr: {stderr_tail}\nstdout: {stdout_tail}"
            )
            return False, errors

        # 4. Check metrics.json exists and has required fields
        metrics_path = idea_dir / "metrics.json"
        if not metrics_path.exists():
            errors.append("Train script did not write metrics.json")
            return False, errors

        try:
            metrics = json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"metrics.json is not valid JSON: {e}")
            return False, errors

        if "status" not in metrics:
            errors.append("metrics.json missing 'status' field")

        primary = cfg.get("report", {}).get("primary_metric", "")
        if primary and primary not in metrics:
            errors.append(
                f"metrics.json missing primary metric '{primary}'. "
                f"Keys present: {list(metrics.keys())}"
            )

        # 5. Check config was received (verify sentinel)
        log_path = idea_dir / "train_output.log"
        if log_path.exists():
            log_text = log_path.read_text()
            if _SENTINEL_VALUE not in log_text:
                errors.append(
                    f"Sentinel key '{_SENTINEL_KEY}' not found in train output. "
                    f"idea_config.yaml may not be reaching the train script."
                )

        # 6. Check for anomalies in metrics
        wer_keys = [k for k in metrics if k.startswith("wer_")]
        if wer_keys:
            wer_values = [metrics[k] for k in wer_keys if isinstance(metrics[k], (int, float))]
            if wer_values and "avg_wer" in metrics:
                expected_avg = sum(wer_values) / len(wer_values)
                actual_avg = metrics["avg_wer"]
                if abs(expected_avg - actual_avg) > 0.5:
                    errors.append(
                        f"avg_wer ({actual_avg}) does not match mean of per-dataset WERs "
                        f"({expected_avg:.2f}). Possible metric computation bug."
                    )

        if errors:
            return False, errors

        logger.info("[SMOKE TEST] PASSED in %.1fs", elapsed)
        return True, []

    except subprocess.TimeoutExpired:
        errors.append(f"Train script timed out after {_TIMEOUT}s on a single sample")
        return False, errors
    except Exception as e:
        errors.append(f"Smoke test error: {e}")
        return False, errors
    finally:
        # Clean up
        if idea_dir.exists():
            shutil.rmtree(idea_dir, ignore_errors=True)
