"""C1 draft regression: retained launch metadata, not mutable cleanup cfg.

Real B1 completion publishes an immutable registered artifact. Only its
existing native fixture substitutes process/GPU boundaries. The initial
artifact store is legitimately nested under this task, not a fake marker.
"""
from pathlib import Path

from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine.scheduler import run_cleanup
from test_native_training_caller_boundaries import case
from test_native_artifact_publication import _declare, _launch, _complete


def test_cleanup_preserves_launch_bound_artifact_root_after_cfg_location_change(case):
    c = case
    _declare(c)
    old_control = c.folder / "control"
    c.cfg.update(_orze_dir=str(old_control), results_dir=str(c.results),
                 idea_lake_db=str(c.lake.db_path))
    tp = _launch(c)
    (c.folder / "model.bin").write_bytes(b"registered immutable artifact")
    assert len(_complete(c, tp)) == 1
    row = current_attempt(c.lake.conn, c.idea, "training")
    assert row["binding"]["artifact_publication"]["root"] == str(old_control / "artifacts")
    records = artifacts_for_attempt(c.lake.conn, tp.attempt_ref)
    assert len(records) == 1
    artifact = Path(records[0]["path"])
    before = artifact.read_bytes()
    disposable = c.folder / "scratch.tmp"
    disposable.write_bytes(b"closed task disposable")
    c.cfg["_orze_dir"] = str(c.results.parent / "new-control-location")
    c.cfg["cleanup"] = {"patterns": ["**/*"]}

    assert run_cleanup(c.results, c.cfg) is None

    assert artifact.is_file(), "cleanup deleted an artifact still bound by the accepted terminal"
    assert artifact.read_bytes() == before
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == records
    assert current_attempt(c.lake.conn, c.idea, "training") == row
    assert not disposable.exists()
