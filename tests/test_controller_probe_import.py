"""New probe draft regression: public lease import must not recurse on itself."""
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

import pytest


@pytest.mark.parametrize("first", ["orze.core.gpu_lease", "orze.hardware.gpu",
                                   "orze.reporting.state", "orze.cli"])
def test_fresh_interpreter_can_import_gpu_lease_before_engine(first):
    root = Path(__file__).resolve().parents[1]
    child = subprocess.Popen([sys.executable, "-c", "import " + first + "; " +
        "from orze.core.gpu_lease import gpu_execution_lease; "
        "from orze.engine.controller_probe import run_probe; "
        "assert callable(gpu_execution_lease) and callable(run_probe)"],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    descriptor = os.pidfd_open(child.pid)
    try:
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stdout + stderr
    finally:
        if not select.select([descriptor], [], [], 0)[0]:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        child.wait(timeout=5)
        os.close(descriptor)
