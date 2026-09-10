"""Real legacy resume evidence must pass the native READY-to-GO comparison.

The old interruption is explicitly unbound compatibility input. Admission,
checkpoint validation, native claim, supervised CPU launch and request
consumption are real; only accelerator allocation/telemetry is replaced.
"""
import contextlib
import json
from pathlib import Path
import sys

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher, scheduler
from orze.engine.resume import admit_resume, prepare_resume_launch
from orze.idea_lake import IdeaLake
from test_resume import resume_case, _write_valid_receipt


def test_valid_legacy_resume_request_starts_native_cpu_worker(resume_case, monkeypatch):
    project, results, folder, checkpoint, cfg, _ = resume_case
    marker = project / "resumed-worker-ran"
    script = Path(cfg["train_script"])
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "assert sys.argv[sys.argv.index('--resume-from')+1] == " + repr(str(checkpoint)) + "\n"
        "Path(" + repr(str(marker)) + ").write_text('resumed')\n"
        "Path(" + repr(str(folder / "metrics.json")) + ").write_text('{\"status\":\"COMPLETED\"}')\n",
        encoding="utf-8")
    base, ideas = project / "base.yaml", project / "ideas.md"
    base.write_text("{}\n", encoding="utf-8")
    ideas.write_text("", encoding="utf-8")
    cfg.update({"base_config": str(base), "ideas_file": str(ideas),
                "python": sys.executable, "_orze_dir": str(project / "control"),
                "timeout": 10, "max_fix_attempts": 0})
    _write_valid_receipt(resume_case)
    admit_resume(folder.name, results, cfg, str(checkpoint))
    lake = IdeaLake(project / "native.db")
    cfg["idea_lake_db"] = str(lake.db_path)
    handles = []
    real_prepare = launcher.prepare_supervised

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        handles.append(process)
        assert process.poll() is None and not marker.exists()
        return process

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    monkeypatch.setattr(launcher, "gpu_execution_lease", lambda *a, **k: contextlib.nullcontext(()))
    monkeypatch.setattr(launcher, "_verify_gpu_free", lambda *a, **k: None)
    tp = None
    try:
        lake.insert(folder.name, "Legacy request, native resume", "lr: 0.001\n", "", status="queued")
        assert scheduler.claim(folder.name, results, 4, lake=lake)
        prepared = prepare_resume_launch(folder.name, results, cfg)
        assert isinstance(prepared["request_path"], Path)
        caught = None
        try:
            tp = launcher.launch(folder.name, 4, results, cfg, lake=lake)
        except launcher.LaunchIntegrityError as exc:
            caught = exc
        assert handles, "must reach the real blocked READY worker"
        assert tp is not None, f"valid prepared resume was rejected: {caught}"
        assert tp.process.wait(timeout=5) == 0
        assert marker.read_text() == "resumed"
        assert not (folder / "resume_request.json").exists()
        assert (folder / "resume_request.consumed.json").exists()
        claim = json.loads((folder / "claim.json").read_bytes())
        assert claim["resume_checkpoint"] == str(checkpoint)
        row = current_attempt(lake.conn, folder.name, "training")
        assert row["state"] == "RUNNING"
        assert row["binding"]["supervision"] == tp.process.binding
        assert row["binding"]["process_pid"] == tp.process.pid
        assert lake.get_fsm_state(folder.name) == "IN_PROGRESS"
    finally:
        for process in handles:
            process.stop(timeout=3)
        if tp is not None:
            tp.close_log()
        lake.close()
