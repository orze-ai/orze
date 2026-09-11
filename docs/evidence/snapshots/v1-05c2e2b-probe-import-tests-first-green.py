"""New probe draft regression: public lease import must not recurse on itself."""
import os
from pathlib import Path
import subprocess
import sys


def test_fresh_interpreter_can_import_gpu_lease_before_engine():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c",
        "from orze.core.gpu_lease import gpu_execution_lease; "
        "from orze.engine.controller_probe import run_probe; "
        "assert callable(gpu_execution_lease) and callable(run_probe)"],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
