"""Tests for determinism hardening win #4 (GPU validation) and the
eval silent-skip instrumentation."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from orze.engine.evaluator import _record_eval_audit, launch_eval, run_post_scripts
from orze.engine.launcher import (
    GpuUnavailableError,
    _verify_gpu_free,
)


# ---------------------------------------------------------------------------
# #4 — pre-launch GPU validation
# ---------------------------------------------------------------------------

def test_verify_gpu_free_passes_when_gpu_has_room():
    """nvidia-smi reports GPU 0 has 40 GiB free — verify lets it through."""
    fake = {0: (1000, 41000)}  # used_mib, total_mib → 40 GiB free
    with patch("orze.engine.gpu_slots._query_all_gpu_usage", return_value=fake):
        _verify_gpu_free(0, min_free_mib=2000)  # no raise


def test_verify_gpu_free_raises_when_under_threshold():
    """GPU 0 reports only 500 MiB free — needs >= 2000 → raise."""
    fake = {0: (40500, 41000)}
    with patch("orze.engine.gpu_slots._query_all_gpu_usage", return_value=fake):
        with pytest.raises(GpuUnavailableError, match="500 MiB free"):
            _verify_gpu_free(0, min_free_mib=2000)


def test_verify_gpu_free_raises_when_gpu_not_reported():
    """nvidia-smi doesn't see GPU 3 — likely offline or masked → raise."""
    fake = {0: (1000, 41000), 1: (1000, 41000)}  # no GPU 3
    with patch("orze.engine.gpu_slots._query_all_gpu_usage", return_value=fake):
        with pytest.raises(GpuUnavailableError, match="not visible"):
            _verify_gpu_free(3, min_free_mib=2000)


def test_verify_gpu_free_fails_open_on_nvidia_smi_failure():
    """Transient nvidia-smi error must NOT block a healthy launch."""
    with patch("orze.engine.gpu_slots._query_all_gpu_usage", return_value={}):
        _verify_gpu_free(0, min_free_mib=2000)  # no raise


def test_verify_gpu_free_noop_when_threshold_zero():
    """min_free_mib=0 disables the check (back-compat for old configs)."""
    # Even if nvidia-smi would report 0 free, threshold 0 means skip.
    _verify_gpu_free(0, min_free_mib=0)
    _verify_gpu_free(-1, min_free_mib=2000)  # gpu<0 also skipped
    _verify_gpu_free(None, min_free_mib=2000)


# ---------------------------------------------------------------------------
# Eval silent-skip instrumentation
# ---------------------------------------------------------------------------

def _make_idea_dir(tmp_path: Path, idea_id: str) -> Path:
    d = tmp_path / idea_id
    d.mkdir()
    return d


def test_eval_skip_records_audit_and_logs_structured(tmp_path, caplog):
    """Pre-existing eval_report.json must cause launch_eval to log
    [EVAL_SKIP] AND append a JSONL audit row (was previously a debug
    log with no persistent trace)."""
    caplog.set_level(logging.INFO, logger="orze")
    idea_id = "idea-eval-skip"
    idea_dir = _make_idea_dir(tmp_path, idea_id)
    (idea_dir / "eval_report.json").write_text('{"wer": 0.05}')

    cfg = {"eval_script": "/usr/bin/true"}
    result = launch_eval(idea_id, gpu=0, results_dir=tmp_path, cfg=cfg)

    assert result is None
    assert "[EVAL_SKIP]" in caplog.text
    assert "reason=output_exists" in caplog.text
    audit = idea_dir / "_eval_audit.jsonl"
    assert audit.exists()
    lines = audit.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["action"] == "skip"
    assert entry["reason"] == "output_exists"
    assert "ts" in entry


def test_post_script_skip_records_audit_and_logs_structured(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="orze")
    idea_id = "idea-ps-skip"
    idea_dir = _make_idea_dir(tmp_path, idea_id)
    # COMPLETED training so run_post_scripts proceeds past status guard
    (idea_dir / "metrics.json").write_text('{"status": "COMPLETED"}')
    (idea_dir / "ps_out.json").write_text('{"x": 1}')

    cfg = {
        "post_scripts": [
            {"script": "/usr/bin/true", "name": "my_ps",
             "output": "ps_out.json"},
        ],
    }
    run_post_scripts(idea_id, gpu=0, results_dir=tmp_path, cfg=cfg)

    assert "[POST_SCRIPT_SKIP]" in caplog.text
    assert "script=my_ps" in caplog.text
    audit = idea_dir / "_eval_audit.jsonl"
    assert audit.exists()
    entry = json.loads(audit.read_text().strip().splitlines()[0])
    assert entry["action"] == "skip"
    assert entry["reason"] == "output_exists"
    assert entry["script"] == "my_ps"


def test_record_eval_audit_appends_multiple_entries(tmp_path):
    """Multiple skips on the same idea must accumulate (JSONL append)."""
    idea_dir = tmp_path / "idea"
    idea_dir.mkdir()
    _record_eval_audit(idea_dir, "skip", "output_exists", path="a")
    _record_eval_audit(idea_dir, "skip", "output_exists", path="b")
    lines = (idea_dir / "_eval_audit.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    e1, e2 = json.loads(lines[0]), json.loads(lines[1])
    assert e1["path"] == "a"
    assert e2["path"] == "b"


def test_record_eval_audit_swallows_io_errors(tmp_path):
    """audit must never block the eval pipeline — bad dir → silent no-op."""
    # Directory does not exist; should not raise.
    _record_eval_audit(tmp_path / "does_not_exist", "skip", "x")
