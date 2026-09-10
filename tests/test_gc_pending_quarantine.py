"""Draft regression: a source-less unconfirmed GC remains observable HOLD."""
import os

from orze.agents.orze_gc import run_gc


def test_next_gc_reports_pending_quarantine_even_after_source_is_detached(tmp_path, monkeypatch):
    results, checkpoints = tmp_path / "results", tmp_path / "checkpoints"
    task = results / "idea-disposable"
    task.mkdir(parents=True)
    (task / "metrics.json").write_bytes(b'{"status":"FAILED"}')
    source = checkpoints / task.name
    source.mkdir(parents=True)
    (source / "weights.pt").write_bytes(b"retain on failed reclamation")
    original = os.unlink
    def denied(path, *args, **kwargs):
        if path == "weights.pt":
            raise PermissionError("selected quarantine payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "unlink", denied)
    cfg = {"_project_root": str(tmp_path), "results_dir": str(results)}
    kwargs = dict(results_dir=results, checkpoints_dir=checkpoints,
                  primary_metric="", keep_top=0, keep_recent=0, cfg=cfg)
    first = run_gc(**kwargs)
    assert first["blocked"] and first["checkpoints"]["deleted"] == 0
    assert not source.exists()
    pending = list((checkpoints / "_orze_gc_quarantine").glob("*/*/content/weights.pt"))
    assert len(pending) == 1 and pending[0].is_file()

    second = run_gc(**kwargs)

    assert second.get("blocked") is True, "missing source hid unresolved quarantine"
    assert second["checkpoints"]["errors"] > 0
    assert second["checkpoints"]["deleted"] == 0
    assert pending[0].read_bytes() == b"retain on failed reclamation"
