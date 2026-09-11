"""Independent actual CLI rejection at READY and terminal publication seams.

Only GPU/provider/legacy entry prohibitions from the existing project fixture
are doubles. Both domain workers and the CLI/Lake/claims/budget/supervision are
real. Hooks transparently call real preparation or artifact registration first.
All replaced files/SQL belong to this test's temporary project.
"""
import json
from pathlib import Path
import sqlite3

import pytest

from orze.engine import native_cpu_action as native
from test_cpu_product_loop import project
from test_cpu_domain_product import submit, request


@pytest.fixture
def analysis_project(project, monkeypatch):
    root, cfg, run = project
    cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}
    submit(root, "idea-source", request("from pathlib import Path; Path('input').write_text('17')",
        outputs={"input": {"path": "input", "max_bytes": 128}}))
    assert run() == 0
    with sqlite3.connect(root / "lake.db") as conn:
        source = json.loads(conn.execute("SELECT record_json FROM research_artifacts").fetchone()[0])
        terminal = json.loads(conn.execute("SELECT terminal_json FROM execution_attempts").fetchone()[0])
    assert terminal["outcome"] == "completed"
    assert terminal["artifact_ids"] == [source["artifact_id"]]
    assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    cfg["action_domain"]["kind"] = "json_observations"
    program = """import json,os
from pathlib import Path
Path('executed').write_text('GO reached')
fds=json.loads(os.environ['ORZE_ACTION_SOURCE_FDS'])
values=[int(os.read(fd,128)) for fd in fds.values()]
assert values==[17]
Path('result.json').write_text(json.dumps({'version':1,'observations':[
 {'name':'value','values':{'value':sum(values)},
  'validation':{'status':'unknown','reason_code':'adapter_reported'},'comparison_scope':None}]}))
"""
    submit(root, "idea-analysis", request(program, sources=[source["artifact_id"]], observation=True,
        outputs={"result": {"path": "result.json", "max_bytes": 4096}}))
    processes = []
    actual_prepare = native.prepare_supervised
    def capture(*args, **kwargs):
        process = actual_prepare(*args, **kwargs)
        processes.append(process)  # Retain actual own handle before any fault.
        return process
    monkeypatch.setattr(native, "prepare_supervised", capture)
    try:
        yield root, cfg, run, source, processes
    finally:
        for process in processes:
            if process.poll() is None:
                process.stop(timeout=0.2)
            assert type(process.poll()) is int


def _held_without_publication(root):
    with sqlite3.connect(root / "lake.db") as conn:
        row = conn.execute("SELECT state,terminal_json FROM execution_attempts "
                           "WHERE task_id='idea-analysis'").fetchone()
        assert row[0] in ("LAUNCHING", "RUNNING") and row[1] is None
        assert conn.execute("SELECT state FROM cpu_action_reservations WHERE task_id='idea-analysis'").fetchall() == [("BOUND",)]
        assert conn.execute("SELECT state FROM cpu_action_reservations WHERE task_id='idea-source'").fetchall() == [("SETTLED",)]
        assert conn.execute("SELECT COUNT(*) FROM research_artifacts WHERE producer_task_id='idea-analysis'").fetchone()[0] == 0
        if conn.execute("SELECT name FROM sqlite_master WHERE name='research_observations'").fetchone():
            assert conn.execute("SELECT COUNT(*) FROM research_observations WHERE evaluator_task_id='idea-analysis'").fetchone()[0] == 0
        assert conn.execute("SELECT current_state FROM idea_state WHERE idea_id='idea-analysis'").fetchone()[0] != "COMPLETE"


def test_cli_ready_source_inode_replacement_refuses_go_and_keeps_reservation(analysis_project, monkeypatch):
    root, _, run, source, processes = analysis_project
    actual_prepare = native.prepare_supervised
    replacements = []
    def replace_after_ready(*args, **kwargs):
        process = actual_prepare(*args, **kwargs)
        assert kwargs["identity"]["attempt_ref"]["task_id"] == "idea-analysis"
        path = Path(source["path"])
        original = path.stat()
        content = path.read_bytes()
        saved = path.with_name("original-content")
        path.rename(saved)
        path.write_bytes(content)
        replacements.append((original.st_dev, original.st_ino, path.stat().st_dev, path.stat().st_ino))
        return process
    monkeypatch.setattr(native, "prepare_supervised", replace_after_ready)
    assert run() == 75
    assert len(replacements) == len(processes) == 1
    assert replacements[0][:2] != replacements[0][2:]
    assert Path(source["path"]).read_bytes() == Path(source["path"]).with_name("original-content").read_bytes()
    assert not list((root / "results" / "idea-analysis").rglob("executed"))
    assert processes[0].closure_receipt()["stop_requested"] is True
    _held_without_publication(root)


def test_cli_terminal_trigger_changes_real_input_after_analysis_registration_and_rolls_back(analysis_project, monkeypatch):
    root, _, run, source, processes = analysis_project
    original_register = native.register_artifacts
    observed = []
    def register_and_install(conn, ref, records):
        result = original_register(conn, ref, records)
        if ref.task_id == "idea-analysis":
            def record_actual(artifact, observation, original_input):
                observed.append(tuple(json.loads(value) for value in (artifact, observation, original_input)))
                return 0
            conn.create_function("observe_analysis_registered", 3, record_actual)
            # The real terminal update fires only after both real registrations.
            conn.execute("""CREATE TRIGGER cpu_analysis_input_after_terminal
                AFTER UPDATE OF state ON execution_attempts
                WHEN NEW.state='TERMINAL' AND NEW.task_id='idea-analysis'
                BEGIN
                  SELECT observe_analysis_registered(
                    (SELECT record_json FROM research_artifacts WHERE producer_attempt_id=NEW.attempt_id),
                    (SELECT record_json FROM research_observations WHERE evaluator_attempt_id=NEW.attempt_id),
                    (SELECT record_json FROM research_artifacts WHERE producer_task_id='idea-source'));
                  UPDATE research_artifacts SET record_json=json_set(record_json,'$.content_sha256',
                    '0000000000000000000000000000000000000000000000000000000000000000')
                    WHERE producer_task_id='idea-source';
                END""")
        return result
    monkeypatch.setattr(native, "register_artifacts", register_and_install)
    assert run() == 75
    assert len(observed) == len(processes) == 1
    artifact, observation, original_input = observed[0]
    assert artifact["producer"]["task_id"] == observation["evaluator"]["task_id"] == "idea-analysis"
    assert observation["result_artifact_ids"] == [artifact["artifact_id"]]
    assert observation["input_artifact_ids"] == [source["artifact_id"]]
    assert original_input == source
    assert processes[0].closure_receipt()["worker_returncode"] == 0
    assert processes[0].closure_receipt()["stop_requested"] is False
    _held_without_publication(root)
    with sqlite3.connect(root / "lake.db") as conn:
        assert json.loads(conn.execute("SELECT record_json FROM research_artifacts WHERE artifact_id=?",
                                       (source["artifact_id"],)).fetchone()[0]) == source
    effects = root / "results" / "idea-analysis" / "_execution_effects"
    assert len(list(effects.glob("*/prepared.json"))) == 1
    assert list(effects.glob("*/committed.json")) == []
