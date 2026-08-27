import subprocess
import sys
import time

import pytest

from orze.core.config import _validate_config
from orze.engine.process import (
    process_tree_cpu_ticks,
    progress_paths_fingerprint,
)


def test_process_tree_cpu_ticks_detect_silent_computation():
    proc = subprocess.Popen([
        sys.executable,
        "-c",
        "import time\nend=time.time()+3\nx=0\n"
        "while time.time()<end: x=(x+1)%1000003\n",
    ])
    try:
        initial = process_tree_cpu_ticks(proc.pid)
        assert initial is not None
        deadline = time.time() + 2
        observed = initial
        while time.time() < deadline and observed <= initial:
            time.sleep(0.02)
            observed = process_tree_cpu_ticks(proc.pid)
            assert observed is not None
        assert observed > initial
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_declared_output_fingerprint_uses_metadata_only(tmp_path):
    output = tmp_path / "decision.md"
    before = progress_paths_fingerprint([output])
    output.write_text("decision", encoding="utf-8")
    after = progress_paths_fingerprint([output])

    assert before != after
    assert progress_paths_fingerprint([]) is None


@pytest.mark.parametrize("field,value", [
    ("stall_minutes", -1),
    ("stall_minutes", True),
    ("stall_minutes", float("nan")),
    ("stall_warmup_seconds", float("inf")),
    ("stall_warmup_seconds", "sixty"),
])
def test_invalid_role_progress_policy_is_rejected(field, value):
    errors, _ = _validate_config({
        "roles": {
            "worker": {
                "mode": "script",
                "script": "worker.py",
                field: value,
            },
        },
    })
    assert any(f"roles.worker.{field}" in error for error in errors)
