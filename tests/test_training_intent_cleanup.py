"""Real launch must close local pipes without releasing uncertain intent."""
import os
import sqlite3
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import current_attempt, ensure_schema
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.execution_identity import _registry_root
from orze.engine.training_attempts import begin
from test_model_lineage import _config
from test_native_training_caller_boundaries import case, _launch, _terminals


class BeginConnection(sqlite3.Connection):
    fault = None

    def commit(self):
        if self.fault == "commit_before":
            raise sqlite3.OperationalError("synthetic native intent commit failure")
        super().commit()
        if self.fault == "commit_after":
            raise sqlite3.OperationalError("synthetic native intent commit response lost")


@pytest.mark.parametrize("fault", ["clean_rejection", "commit_before", "commit_after", "existing_intent"])
def test_begin_failure_closes_only_local_resources_without_resolving_intent(case, monkeypatch, fault):
    c = case
    c.cfg.update(_config(c.results.parent))
    # Only the namespace capability probe is simulated; lineage policy,
    # manifests, fd allocation, identity registry and begin are all real.
    monkeypatch.setattr("orze.engine.launcher._probe_kernel_boundary", lambda **kw: None)
    fds = []
    original_pipe = os.pipe

    def capture_pipe():
        pair = original_pipe()
        fds.extend(pair)
        return pair

    monkeypatch.setattr(os, "pipe", capture_pipe)
    c.lake.conn.execute("BEGIN IMMEDIATE")
    ensure_schema(c.lake.conn)
    c.lake.conn.commit()
    if fault == "clean_rejection":
        c.lake.conn.execute("CREATE TRIGGER reject_intent BEFORE INSERT ON execution_attempts "
                            "BEGIN SELECT RAISE(IGNORE); END")
        c.lake.conn.commit()
    elif fault == "existing_intent":
        import json
        claim = json.loads((c.folder / "claim.json").read_text())
        begin(c.lake, SimpleNamespace(idea_id=c.idea, attempt_id=claim["attempt_id"], gpu=0), c.folder)
    else:
        c.lake.conn.close()
        c.lake.conn = sqlite3.connect(c.lake.db_path, factory=BeginConnection)
        c.lake.conn.row_factory = sqlite3.Row
        c.lake.conn.fault = fault

    try:
        with pytest.raises((AttemptEffectBusy, AttemptEffectInDoubt)):
            _launch(c)
        assert len(fds) == 2, "actual lineage launch must allocate its private pipe"
        assert c.popen_calls == c.stops == []
        for fd in fds:
            with pytest.raises(OSError):
                os.fstat(fd)
        row = current_attempt(c.lake.conn, c.idea, "training")
        registry = list(_registry_root(c.results, c.cfg).glob("*.json"))
        if fault == "clean_rejection":
            assert row is None and registry == []
            assert not (c.folder / "_attempt_effect.lock").exists()
        elif fault == "existing_intent":
            assert row["state"] == "LAUNCHING" and len(registry) == 1
        else:
            assert (row is None if fault == "commit_before" else row["state"] == "LAUNCHING")
            assert len(registry) == 1
            assert (c.folder / "_attempt_effect.lock").is_dir()
        assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
        assert not _terminals(c)
        assert not (c.folder / "train_output.log").exists()
        assert not (c.folder / "metrics.json").exists()
    finally:
        # Baseline intentionally leaks descriptors; don't leak the test runner.
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
