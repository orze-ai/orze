"""Startup smoke test — verify train script contract before burning GPU hours.

CALLING SPEC:
    from orze.engine.smoke_test import run_smoke_test

    passed, errors = run_smoke_test(cfg, results_dir)
        cfg: dict — orze config (needs train_script, base_config, ideas_file)
        results_dir: Path
        -> (True, []) on success
        -> (False, [error_strings]) on failure

Runs a single-sample test idea through the pipeline:
  1. Writes idea_config.yaml with a sentinel key
  2. Launches train.py on CPU with max_samples=1
  3. Verifies: config flows through, metrics.json written, primary metric present
  4. Cleans up

Disable with smoke_test: false in orze.yaml.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import yaml
from pathlib import Path

logger = logging.getLogger("orze")

_SMOKE_ID = "_smoke_test"
_SENTINEL_KEY = "_smoke_sentinel"
_SENTINEL_VAL = "orze_contract_check_42"
_TIMEOUT = 300


def _find_free_gpu(cfg: dict):
    """Find a GPU with enough free memory, or return None."""
    try:
        import subprocess as _sp
        result = _sp.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        threshold = cfg.get("gpu_mem_threshold", 40000)
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 2:
                gpu_id, free_mb = int(parts[0].strip()), int(parts[1].strip())
                if free_mb > threshold:
                    return gpu_id
    except Exception:
        pass
    return None


def run_smoke_test(cfg: dict, results_dir: Path) -> tuple:
    """Run end-to-end smoke test. Returns (passed, errors)."""
    errors = []
    idea_dir = results_dir / _SMOKE_ID

    if idea_dir.exists():
        shutil.rmtree(idea_dir, ignore_errors=True)

    try:
        idea_dir.mkdir(parents=True, exist_ok=True)

        # Write idea_config.yaml with sentinel + minimal eval
        idea_cfg = {_SENTINEL_KEY: _SENTINEL_VAL, "max_samples": 1}
        base_path = cfg.get("base_config", "")
        if Path(base_path).exists():
            base = yaml.safe_load(Path(base_path).read_text()) or {}
            # Use first eval task only for speed
            tasks = base.get("eval_tasks", [])
            if tasks:
                idea_cfg["eval_tasks"] = [tasks[0]]

        (idea_dir / "idea_config.yaml").write_text(
            yaml.dump(idea_cfg, default_flow_style=False))

        # Launch train.py on CPU
        cmd = [
            cfg.get("python", sys.executable),
            cfg["train_script"],
            "--idea-id", _SMOKE_ID,
            "--results-dir", str(results_dir),
            "--ideas-md", cfg.get("ideas_file", "ideas.md"),
            "--config", base_path,
        ]
        env = dict(os.environ)
        # Try to find a free GPU; fall back to CPU
        free_gpu = _find_free_gpu(cfg)
        if free_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
            logger.info("[SMOKE] Running 1-sample test on GPU %d...", free_gpu)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("[SMOKE] Running 1-sample test on CPU (no free GPU)...")
        t0 = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT, env=env)
        elapsed = time.time() - t0

        # Check exit code
        if result.returncode != 0:
            errors.append(
                f"Exit code {result.returncode}. "
                f"stderr: {(result.stderr or '')[-300:]}")
            return False, errors

        # Check metrics.json
        mp = idea_dir / "metrics.json"
        if not mp.exists():
            errors.append("No metrics.json written")
            return False, errors

        try:
            metrics = json.loads(mp.read_text())
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"Invalid metrics.json: {e}")
            return False, errors

        if "status" not in metrics:
            errors.append("metrics.json missing 'status'")

        primary = (cfg.get("report") or {}).get("primary_metric", "")
        if primary and primary not in metrics:
            errors.append(f"Missing primary metric '{primary}' in metrics.json")

        # Check sentinel reached the train script
        log = idea_dir / "train_output.log"
        if log.exists() and _SENTINEL_VAL not in log.read_text():
            errors.append(
                "Sentinel not found in output — "
                "idea_config.yaml may not reach the train script")

        # Validate metric consistency
        from orze.engine.guardrails import validate_avg_metric
        if primary:
            warn = validate_avg_metric(metrics, primary)
            if warn:
                errors.append(f"Metric check: {warn}")

        if errors:
            return False, errors

        logger.info("[SMOKE] PASSED in %.1fs", elapsed)
        return True, []

    except subprocess.TimeoutExpired:
        errors.append(f"Timed out after {_TIMEOUT}s")
        return False, errors
    except Exception as e:
        errors.append(f"Error: {e}")
        return False, errors
    finally:
        if idea_dir.exists():
            shutil.rmtree(idea_dir, ignore_errors=True)
