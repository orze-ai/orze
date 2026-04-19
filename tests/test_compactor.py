"""Tests for F5 — rolling-window compaction of prompt files."""
from pathlib import Path

from orze.engine.compactor import compact_file, compact_standard_paths


def _cycles(n: int, per_cycle_bytes: int = 1000) -> str:
    parts = []
    for i in range(1, n + 1):
        body = "content line\n" * (per_cycle_bytes // 15)
        parts.append(f"## cycle {i}\n{body}\n")
    return "".join(parts)


def test_compact_file_keeps_last_50_cycles(tmp_path):
    p = tmp_path / "_retrospection.txt"
    p.write_text(_cycles(200), encoding="utf-8")
    size_before = p.stat().st_size
    assert size_before > 150_000

    full = _cycles(200)
    last_50_cycles = full[full.index("## cycle 151\n"):]

    summary = compact_file(p, keep_last=50, hard_max_bytes=150_000)
    assert summary["cycles_kept"] == 50
    assert summary["cycles_summarized"] == 150

    new_text = p.read_text(encoding="utf-8")
    assert "<SUMMARY:" in new_text
    assert new_text.endswith(last_50_cycles)
    # Keep-N guarantee preserves last_50 even if that exceeds hard cap,
    # but in this test sizing, new file IS under hard cap.
    assert summary["bytes_after"] <= 150_000


def test_compact_file_noop_under_threshold(tmp_path):
    p = tmp_path / "small.txt"
    p.write_text("## cycle 1\nhi\n## cycle 2\nbye\n", encoding="utf-8")
    before = p.read_text()
    summary = compact_file(p, keep_last=50, hard_max_bytes=150_000)
    assert summary["mode"] == "noop"
    assert p.read_text() == before


def test_compact_file_no_cycle_header_tail_truncates(tmp_path):
    p = tmp_path / "flat.txt"
    # 500 KB, no cycle delimiters
    p.write_text("x" * 500_000, encoding="utf-8")
    summary = compact_file(p, hard_max_bytes=150_000)
    assert summary["mode"].startswith("tail")
    assert p.stat().st_size <= 150_000


def test_compact_standard_paths(tmp_path):
    (tmp_path / "_retrospection.txt").write_text(_cycles(100),
                                                 encoding="utf-8")
    (tmp_path / "_skill_composed_research.md").write_text(_cycles(100),
                                                          encoding="utf-8")
    results = compact_standard_paths(tmp_path)
    assert len(results) >= 2
    for r in results:
        if r["bytes_before"] > 150_000:
            assert r["bytes_after"] <= 150_000
