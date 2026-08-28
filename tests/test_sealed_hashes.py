from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from orze.core.config import _validate_config, load_project_config
from orze.engine.phases import OrzePhaseMixin


def test_pinned_hash_path_is_automatically_sealed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "eval_manifest.json"
    source.write_text('{"revision":"abc"}', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    config = tmp_path / "orze.yaml"
    config.write_text(
        "auto_seal_eval: false\n"
        "sealed_hashes:\n"
        f"  {source}: {digest}\n",
        encoding="utf-8",
    )

    cfg = load_project_config(str(config))

    assert str(source) in cfg["sealed_files"]
    assert cfg["sealed_hashes"][str(source)] == digest


def test_sealed_hashes_rejects_malformed_digest():
    errors, _ = _validate_config({"sealed_hashes": {"eval.py": "not-a-sha"}})
    assert any("64-character SHA-256" in error for error in errors)


def test_sealed_hashes_rejects_non_mapping():
    errors, _ = _validate_config({"sealed_hashes": ["eval.py"]})
    assert "sealed_hashes: must be a mapping of path to SHA-256" in errors


def test_sealed_violation_blocks_training_dispatch(tmp_path, monkeypatch):
    source = tmp_path / "eval_manifest.json"
    source.write_text('{"revision":"drifted"}', encoding="utf-8")
    expected = hashlib.sha256(b'{"revision":"expected"}').hexdigest()
    (tmp_path / ".sealed_hashes").write_text(
        json.dumps({str(source): expected}), encoding="utf-8"
    )
    notices = []
    monkeypatch.setattr("orze.engine.phases.get_gpu_memory_used", lambda _gpu: 0)
    monkeypatch.setattr(
        "orze.engine.phases.notify",
        lambda event, payload, cfg: notices.append((event, payload)),
    )
    runner = SimpleNamespace(
        cfg={"sealed_files": [str(source)], "gpu_mem_threshold": 2000},
        results_dir=tmp_path,
        active_evals={},
        active={},
        gpu_ids=[4, 5, 6, 7],
    )
    unclaimed = ["idea-must-not-launch"]

    free = OrzePhaseMixin._launch_training(runner, unclaimed, True, {})

    assert free == [4, 5, 6, 7]
    assert unclaimed == ["idea-must-not-launch"]
    assert notices[0][0] == "sealed_file_changed"
    assert "blocked" in notices[0][1]["message"]


def test_managed_run_uses_explicit_pin_without_shared_manifest(
        tmp_path, monkeypatch):
    source = tmp_path / "eval_manifest.json"
    source.write_text('{"revision":"expected"}', encoding="utf-8")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    notices = []
    monkeypatch.setattr("orze.engine.phases.get_gpu_memory_used", lambda _gpu: 0)
    monkeypatch.setattr(
        "orze.engine.phases.notify",
        lambda event, payload, cfg: notices.append((event, payload)),
    )
    runner = SimpleNamespace(
        cfg={
            "_managed_idea_id": "idea-managed",
            "sealed_files": [str(source)],
            "sealed_hashes": {str(source): expected},
            "gpu_mem_threshold": 2000,
        },
        results_dir=tmp_path,
        active_evals={},
        active={},
        gpu_ids=[4],
        once=True,
        running=True,
    )

    free = OrzePhaseMixin._launch_training(runner, [], True, {})

    assert free == [4]
    assert notices == []
