"""C1 draft regression: real legacy input declarations are not disposable.

No native artifact contract is enabled, and no catalog/claim exists. The
ordinary public cleanup API must retain declared consumer inputs while still
deleting an unrelated matched scratch file. No provider, child or GPU runs.
"""

from pathlib import Path

import pytest

from orze.core.model_lineage import _idea_path, validate_model_lineage_config
from orze.engine.resume import _project_root, _resolve_path
from orze.engine.scheduler import run_cleanup


@pytest.mark.parametrize("declaration", [
    "model_lineage_artifact",
    "evaluation_checkpoint",
    "resume_immutable_directory",
])
def test_public_cleanup_retains_declared_legacy_consumer_inputs(
        tmp_path, declaration):
    results = tmp_path / "results"
    folder = results / "idea-legacy-input"
    folder.mkdir(parents=True)
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "cleanup": {"patterns": ["**/*"]},
        "gc": {"enabled": False},
    }
    if declaration == "model_lineage_artifact":
        cfg.update(
            model_lineage={"enabled": True, "artifact": "lineage-weights/model.bin"},
            data_boundaries={
                "forbidden_in_training": [str(tmp_path / "private-evaluation-input")],
                "training_network": "deny",
            },
            data_separation={"enabled": True},
        )
        assert validate_model_lineage_config(cfg) == []
        target = _idea_path(folder, cfg["model_lineage"]["artifact"])
        assert target == folder / "lineage-weights" / "model.bin"
        protected = [target]
    elif declaration == "evaluation_checkpoint":
        cfg.update(eval_script="evaluate.py", eval_checkpoint="evaluation-inputs/best.pt")
        # phases' legacy evaluation backlog resolves this declaration at the
        # task directory, independently of project-level script paths.
        target = folder / cfg["eval_checkpoint"]
        protected = [target]
    else:
        target = folder / "resume-inputs"
        cfg["resume"] = {
            "enabled": True,
            "immutable_inputs": [target.relative_to(tmp_path).as_posix()],
        }
        target.mkdir()
        project_root = _project_root(cfg, results)
        # Use the existing resume consumer's actual path resolver: these are
        # project-relative declarations, not task-relative paths.
        assert _resolve_path(
            cfg["resume"]["immutable_inputs"][0],
            project_root, [project_root], "immutable_input") == target
        protected = [target / "dataset.lock", target / "nested" / "weights.bin"]

    before = {}
    for index, path in enumerate(protected):
        path.parent.mkdir(parents=True, exist_ok=True)
        body = ("pinned consumer input " + str(index)).encode("utf-8")
        path.write_bytes(body)
        before[path] = body
    scratch = folder / "scratch.tmp"
    scratch.write_bytes(b"ordinary matched disposable")
    metrics = folder / "metrics.json"
    metrics.write_bytes(b'{"status":"COMPLETED"}')

    assert run_cleanup(results, cfg) is None

    assert not scratch.exists(), "the real cleanup consumer must have run"
    missing = [str(path.relative_to(folder)) for path in before if not path.is_file()]
    assert not missing, "declared legacy inputs were deleted: " + repr(missing)
    assert {path: path.read_bytes() for path in before} == before
    assert metrics.read_bytes() == b'{"status":"COMPLETED"}'
    assert not list(tmp_path.rglob("*.db"))
    assert not (folder / "_execution_catalog.json").exists()
