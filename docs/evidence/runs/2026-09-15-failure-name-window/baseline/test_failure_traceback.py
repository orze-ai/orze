"""Guardrails for failure analysis traceback extraction.

If these fail, failure_analysis.json will contain generic strings instead of
real Python tracebacks — making the admin knowledge base useless for product dev.
"""
import json
from pathlib import Path

from orze.engine.failure import (
    _extract_traceback,
    build_failure_analysis,
    write_failure_analysis,
)


SAMPLE_LOG_WITH_TRACEBACK = """\
Epoch 1/10: loss=2.31
Epoch 2/10: loss=1.85
Traceback (most recent call last):
  File "train.py", line 42, in <module>
    model = build_model(cfg)
  File "model.py", line 18, in build_model
    return ResNet(layers=cfg["layers"])
KeyError: 'layers'
"""

SAMPLE_LOG_NO_TRACEBACK = """\
Epoch 1/10: loss=2.31
Epoch 2/10: loss=1.85
CUDA error: out of memory
Killed
"""


class TestExtractTraceback:
    def test_extracts_real_traceback(self):
        result = _extract_traceback(SAMPLE_LOG_WITH_TRACEBACK)
        assert "Traceback (most recent call last)" in result
        assert "KeyError: 'layers'" in result
        assert "train.py" in result

    def test_fallback_returns_tail_lines(self):
        result = _extract_traceback(SAMPLE_LOG_NO_TRACEBACK)
        assert "CUDA error: out of memory" in result
        assert "Killed" in result

    def test_empty_input(self):
        result = _extract_traceback("")
        assert isinstance(result, str)

    def test_truncates_long_traceback(self):
        long_tb = "Traceback (most recent call last):\n" + ("  File 'x.py', line 1\n    pass\n" * 500)
        result = _extract_traceback(long_tb)
        assert len(result) <= 2000

    def test_picks_last_traceback(self):
        text = (
            "Traceback (most recent call last):\n  File 'a.py'\nFirstError\n\n"
            "Traceback (most recent call last):\n  File 'b.py'\nSecondError\n"
        )
        result = _extract_traceback(text)
        assert "b.py" in result
        assert "SecondError" in result


class TestBuildFailureAnalysis:
    def test_includes_traceback_field(self):
        analysis = build_failure_analysis("crash", "exit code 1", log_tail=SAMPLE_LOG_WITH_TRACEBACK)
        assert "traceback" in analysis
        assert "Traceback (most recent call last)" in analysis["traceback"]

    def test_traceback_from_log_tail_not_error_text(self):
        analysis = build_failure_analysis("crash", "Process failed", log_tail=SAMPLE_LOG_WITH_TRACEBACK)
        assert "KeyError: 'layers'" in analysis["traceback"]

    def test_falls_back_to_error_text(self):
        analysis = build_failure_analysis("oom", "CUDA out of memory\nAllocator failed")
        assert "traceback" in analysis
        assert "CUDA out of memory" in analysis["traceback"]

    def test_required_fields(self):
        analysis = build_failure_analysis("crash", "error")
        for key in ("category", "what", "why", "lesson", "traceback", "timestamp"):
            assert key in analysis, f"Missing required field: {key}"

    def test_what_is_first_line(self):
        analysis = build_failure_analysis("crash", "line1\nline2\nline3")
        assert analysis["what"] == "line1"


class TestWriteFailureAnalysis:
    def test_writes_json_with_traceback(self, tmp_path):
        idea_dir = tmp_path / "idea-001"
        idea_dir.mkdir()
        log = idea_dir / "train_output.log"
        log.write_text(SAMPLE_LOG_WITH_TRACEBACK)

        write_failure_analysis(idea_dir, "crash", "Process failed")

        fa = json.loads((idea_dir / "failure_analysis.json").read_text())
        assert "traceback" in fa
        assert "Traceback (most recent call last)" in fa["traceback"]
        assert "KeyError" in fa["traceback"]

    def test_works_without_log_file(self, tmp_path):
        idea_dir = tmp_path / "idea-002"
        idea_dir.mkdir()

        write_failure_analysis(idea_dir, "oom", "CUDA out of memory")

        fa = json.loads((idea_dir / "failure_analysis.json").read_text())
        assert fa["category"] == "oom"
        assert "traceback" in fa
