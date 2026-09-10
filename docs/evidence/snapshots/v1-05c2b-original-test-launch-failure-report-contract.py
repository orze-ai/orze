"""Exact native report/duplicate controls for the new controller-action API."""
import json
import sqlite3

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.launch_failure_report import report_launch_failure
from test_launch_failure_report import _fail_initialization, _files, _phase
from test_training_launch_termination_handoff import scenario
from test_execution_authority import FaultConnection


def test_duplicate_report_does_not_rewrite_or_recount_and_reprojects_local_state(scenario, monkeypatch):
    runner, child, popen, fixer = scenario
    error = _fail_initialization(monkeypatch, child)
    _phase(runner)
    folder = runner.results_dir / "idea-handoff"
    before = _files(folder)
    history = runner.lake.get_fsm_history(folder.name)
    assert runner.failure_counts == {folder.name: 1}
    for counts in (runner.failure_counts, {}):
        result = report_launch_failure(runner.lake, folder, error, counts, runner.cfg)
        assert result["status"] == "duplicate"
        assert counts == {folder.name: 1}
    assert _files(folder) == before
    assert runner.lake.get_fsm_history(folder.name) == history
    assert popen.call_count == 1
    fixer.assert_not_called()


@pytest.mark.parametrize("missing", ["reference", "lake"])
def test_native_report_never_looks_up_a_missing_authority_token(scenario, monkeypatch, missing):
    runner, child, _, _ = scenario
    error = _fail_initialization(monkeypatch, child)
    _phase(runner)
    folder = runner.results_dir / "idea-handoff"
    before = _files(folder)
    if missing == "reference":
        del error._orze_launch_attempt_ref
    with pytest.raises(AttemptEffectBusy):
        report_launch_failure(None if missing == "lake" else runner.lake,
                              folder, error, {}, runner.cfg)
    assert _files(folder) == before


def test_report_sql_failure_holds_partial_files_and_does_not_publish_count(scenario, monkeypatch):
    runner, child, _, fixer = scenario
    _fail_initialization(monkeypatch, child)
    runner.lake.conn.execute(
        "CREATE TRIGGER refuse_report BEFORE UPDATE ON idea_state "
        "WHEN NEW.current_state='FAILED' BEGIN SELECT RAISE(IGNORE); END")
    runner.lake.conn.commit()
    with pytest.raises(AttemptEffectInDoubt):
        _phase(runner)
    folder = runner.results_dir / "idea-handoff"
    assert runner.failure_counts == {}
    fixer.assert_not_called()
    assert runner.lake.get_fsm_state(folder.name) == "CLAIMED"
    assert current_attempt(runner.lake.conn, folder.name, "launch_failure_report") is None
    assert (folder / "_attempt_effect.lock").is_dir()
    assert json.loads((folder / "metrics.json").read_text())["status"] == "FAILED"


def test_genuinely_legacy_failure_declines_without_creating_anything(tmp_path):
    folder = tmp_path / "idea-legacy"
    assert report_launch_failure(None, folder, RuntimeError("legacy"), {}, {}) is None
    assert not folder.exists()


def test_duplicate_projection_rejects_postcommit_action_receipt_change(scenario, monkeypatch):
    runner, child, _, _ = scenario
    error = _fail_initialization(monkeypatch, child)
    _phase(runner)
    folder = runner.results_dir / "idea-handoff"
    runner.lake.conn.close()
    runner.lake.conn = sqlite3.connect(runner.cfg["idea_lake_db"], factory=FaultConnection)
    runner.lake.conn.row_factory = sqlite3.Row

    def change_receipt():
        runner.lake.conn.execute(
            "UPDATE execution_attempts SET terminal_json=json_set(terminal_json," 
            "'$.failure_count_after',99) WHERE phase='launch_failure_report'")
        sqlite3.Connection.commit(runner.lake.conn)

    runner.lake.conn.after_commit = change_receipt
    counts = {}
    with pytest.raises(AttemptEffectInDoubt):
        report_launch_failure(runner.lake, folder, error, counts, runner.cfg)
    assert counts == {}
    assert (folder / "_attempt_effect.lock").is_dir()
