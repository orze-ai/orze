"""Legacy recovery must not adopt native execution or release native HOLD."""
import os
from unittest.mock import Mock

import pytest

from orze.engine import lifecycle
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt, attempt_effect_lock
from test_native_training_caller_boundaries import case, _launch


def _snapshot(c):
    files = {str(path.relative_to(c.folder)): path.read_bytes()
             for path in c.folder.rglob("*") if path.is_file()}
    tables = {}
    for table in ("ideas", "idea_state", "idea_transitions", "idea_stage_state",
                  "idea_stage_transitions", "execution_attempts"):
        tables[table] = [tuple(row) for row in c.lake.conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid")]
    return files, tables


@pytest.mark.parametrize("entry", ["startup", "dead_pid"])
@pytest.mark.parametrize("held", [False, True])
def test_native_execution_is_not_legacy_adopted_from_dead_pid_or_completed_metrics(case, monkeypatch, entry, held):
    c = case
    c.cfg.update(results_dir=str(c.results), idea_lake_db=str(c.lake.db_path))
    tp = _launch(c)
    tp.close_log()
    (c.folder / "metrics.json").write_text('{"status":"COMPLETED","score":0}')
    for name in ("claim.json", "metrics.json", "train_output.log"):
        os.utime(c.folder / name, (1, 1))
    if held:
        with pytest.raises(AttemptEffectInDoubt):
            with attempt_effect_lock(c.folder):
                raise AttemptEffectInDoubt("synthetic unresolved native publication")
    monkeypatch.setattr(lifecycle, "_running_idea_pids", lambda: set())
    monkeypatch.setattr(lifecycle, "process_is_running", lambda *a: False)
    reaper = Mock(return_value=True)
    monkeypatch.setattr(lifecycle, "terminate_recorded_process_group", reaper)
    before = _snapshot(c)
    if entry == "startup":
        lifecycle.reconcile_stale_running(c.cfg)
    else:
        assert lifecycle.reconcile_running_dead_pids(c.cfg) == 0
    assert _snapshot(c) == before
    reaper.assert_not_called()
