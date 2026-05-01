"""Round-2 F2: ``verified_results`` accepts both legacy path-string and
new list-of-idea-ids forms."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orze.engine.phases import _load_verified  # noqa: E402


def test_legacy_path_string(tmp_path):
    f = tmp_path / "verified_best.json"
    f.write_text(json.dumps({"best_id": "idea-1", "score": 0.9}),
                 encoding="utf-8")
    out = _load_verified(tmp_path, {"verified_results": "verified_best.json"})
    assert out == {"best_id": "idea-1", "score": 0.9}


def test_new_list_of_ids(tmp_path):
    idea = tmp_path / "idea-02e83b"
    idea.mkdir()
    (idea / "full_scale_metrics.json").write_text(
        json.dumps({"wer": 0.05, "epoch": 12}), encoding="utf-8")
    out = _load_verified(
        tmp_path, {"verified_results": ["idea-02e83b", "idea-missing"]})
    assert out == {"idea-02e83b": {"wer": 0.05, "epoch": 12}}


def test_empty_or_none(tmp_path):
    assert _load_verified(tmp_path, {}) is None
    assert _load_verified(tmp_path, {"verified_results": None}) is None


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_legacy_path_string(Path(td))
        print("test_legacy_path_string OK")
    with tempfile.TemporaryDirectory() as td:
        test_new_list_of_ids(Path(td))
        print("test_new_list_of_ids OK")
    with tempfile.TemporaryDirectory() as td:
        test_empty_or_none(Path(td))
        print("test_empty_or_none OK")
