"""Native CPU members through real registered stop, CLI and private children.

These are new opt-in capability requirements, not legacy CPU regressions.
Only optional license discovery and forbidden GPU/provider boundaries are
substituted in child interpreters. Session, supervisor and budget code is real.
"""
import json
import os
from pathlib import Path
import select
import sqlite3
import subprocess
import sys
import time

import pytest
import yaml

PROFILE = {"version": 1, "profile": "local_cpu_stop_v1"}
CORE = Path(__file__).resolve().parents[1]

SITE = '''
import os,sys
if 'orze.cli' in getattr(sys,'orig_argv',()):
    from orze import extensions
    def forbidden(*a,**k):
        raise AssertionError('CPU controller touched GPU/provider/legacy boundary')
    extensions.has_pro=forbidden
    extensions._find_pro_key=forbidden
    try:
        from orze_pro import _gate
        _gate.require_license=lambda:None
    except ImportError:pass
    from orze.hardware import gpu
    gpu.detect_all_gpus=forbidden
    from orze.core import gpu_lease
    gpu_lease.acquire_gpu_leases=forbidden
    gpu_lease.assert_gpu_scope_idle=forbidden
    from orze import lifecycle
    lifecycle.do_stop=forbidden
    if os.environ.get('CPU_STOP_FAULT')=='unsettled':
        from orze.core import cpu_action_budget
        cpu_action_budget.settle=lambda *a,**k:'settled'
    if os.environ.get('CPU_STOP_FAULT')=='orphan_reservation':
        from orze.engine import cpu_phase
        from orze.core import cpu_action_budget
        actual_start=cpu_phase.start
        def start(engine):
            actual_start(engine)
            cpu_action_budget.reserve(engine.lake,engine._cpu_scope,'orphan-0001',1)
        cpu_phase.start=start
    if os.environ.get('CPU_STOP_FAULT')=='close_response_lost':
        from orze.idea_lake import IdeaLake
        actual_close=IdeaLake.close
        def close(self):
            actual_close(self)
            raise OSError('fixture: Lake close response lost')
        IdeaLake.close=close
'''


@pytest.fixture
def cpu_controller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sitecustomize.py").write_text(SITE)
    cfg = {"controller_control": PROFILE,
        "execution": {"version": 1, "resource": "cpu", "slots": 1,
                      "wall_budget_seconds": 120},
        "results_dir": str(tmp_path / "results"), "idea_lake_db": str(tmp_path / "lake.db"),
        "ideas_file": str(tmp_path / "ideas.md"), "min_disk_gb": 0,
        "telemetry": False, "auto_upgrade": False, "max_fix_attempts": 0,
        "metric_harvest": {"enabled": False},
        "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .05}}
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = {k: v for k, v in os.environ.items() if not k.startswith("ORZE_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join([str(tmp_path), str(CORE / "src"), os.environ.get("PYTHONPATH", "")]))
    processes = []
    files = []

    def launch(program=None, *, once=False, fault=None):
        if program is not None:
            action = {"version": 1, "adapter": "command", "purpose": "registered CPU closure",
                "inputs": {}, "command": [sys.executable, "-c", program],
                "timeout_seconds": 60, "outputs": {}}
            (tmp_path / "ideas.md").write_text("## idea-0001: owned CPU action\n\n```yaml\n" +
                yaml.safe_dump({"kind": "native_cpu_action", "action": action}) + "```\n")
        output = (tmp_path / ("controller-%d.log" % len(processes))).open("wb")
        files.append(output)
        proc = subprocess.Popen([sys.executable, "-m", "orze.cli", "-c", str(path)] +
            (["--once"] if once else []), cwd=tmp_path,
            env=dict(env, CPU_STOP_FAULT=fault or ""), stdout=output, stderr=subprocess.STDOUT)
        processes.append(proc)
        return proc

    def wait_for(proc, predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                if predicate():
                    return
            except (sqlite3.OperationalError, FileNotFoundError):
                pass
            if proc.poll() is not None:
                pytest.fail((tmp_path / "controller-0.log").read_text())
            time.sleep(.02)
        pytest.fail("owned fixture did not reach the requested boundary")

    try:
        yield tmp_path, path, cfg, env, launch, wait_for
    finally:
        for proc in processes:
            if proc.poll() is None:
                # Only direct Popen children; no PID-file lookup or adoption.
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            else:
                proc.wait()
        for output in files:
            output.close()


def rows(root, sql):
    with sqlite3.connect(root / "lake.db") as conn:
        return conn.execute(sql).fetchall()


def ack(root):
    raw = rows(root, "SELECT ack_json FROM controller_sessions")[0][0]
    return None if raw is None else json.loads(raw)


@pytest.mark.parametrize("program", [None, "pass"])
def test_cpu_registered_natural_exit_closes_budget_and_members(cpu_controller, program):
    root, path, cfg, env, launch, wait_for = cpu_controller
    proc = launch(program, once=True)
    assert proc.wait(timeout=20) == 0, (root / "controller-0.log").read_text()
    proof = ack(root)
    assert proof["members"]["member_count"] == int(program is not None)
    assert proof["resources"]["gpu_scope"] == []
    assert proof["resources"]["gpu_leases"] == "not_acquired"
    assert proof["resources"]["pid_file"] == "not_created"
    assert proof["resources"]["cpu_budget"]["reservation_count"] == int(program is not None)
    assert proof["resources"]["cpu_budget"]["active_reservations"] == 0
    assert not list((root / "results").glob(".orze.pid*"))
    assert rows(root, "SELECT state FROM cpu_action_reservations") == ([('SETTLED',)] if program else [])
    # Registration is retained; a second ordinary invocation cannot adopt it.
    again = launch(once=True)
    assert again.wait(timeout=15) == 75
    assert rows(root, "SELECT COUNT(*) FROM controller_sessions") == [(1,)]


def test_cpu_registered_stop_closes_worker_and_descendant_before_ack(cpu_controller):
    from orze.core.config import load_project_config
    from orze.engine.controller_session import _Observer
    root, path, cfg, env, launch, wait_for = cpu_controller
    marker = root / "child.json"
    program = ("import os,json,subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])\n"
        f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),child.pid]))\n"
        "time.sleep(120)\n")
    proc = launch(program)
    wait_for(proc, marker.exists)
    child_fds = [os.pidfd_open(pid) for pid in json.loads(marker.read_text())]
    observer = _Observer(load_project_config(str(path)))
    try:
        result = subprocess.run([sys.executable, '-m', 'orze.cli', 'stop', '-c', str(path),
            '--timeout', '15'], cwd=root, env=env, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr + (root / 'controller-0.log').read_text()
        assert proc.wait(timeout=5) == 0
        assert observer.exited()
        assert all(select.select([fd], [], [], 0)[0] for fd in child_fds)
        proof = ack(root)
        observer.verify_drain(proof)
        assert proof['members']['member_count'] == 1
        assert proof['resources']['cpu_budget']['reservation_count'] == 1
        assert rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',)]
        # The observer independently rejects changed settlement after the ACK.
        with sqlite3.connect(root / 'lake.db') as conn:
            conn.execute("UPDATE cpu_action_reservations SET terminal_sha256=?", ('0' * 64,))
        from orze.engine.controller_control import ControllerHOLD
        with pytest.raises(ControllerHOLD, match='cpu'):
            observer.verify_drain(proof)
    finally:
        observer.close()
        for fd in child_fds:
            os.close(fd)


def test_cpu_terminal_without_budget_settlement_cannot_ack(cpu_controller):
    root, path, cfg, env, launch, wait_for = cpu_controller
    proc = launch('pass', once=True, fault='unsettled')
    assert proc.wait(timeout=20) == 75
    assert rows(root, 'SELECT state FROM execution_attempts') == [('TERMINAL',)]
    assert rows(root, 'SELECT state FROM cpu_action_reservations') == [('BOUND',)]
    assert ack(root) is None


def test_cpu_unbound_reservation_without_any_member_cannot_ack(cpu_controller):
    root, path, cfg, env, launch, wait_for = cpu_controller
    proc = launch(once=True, fault='orphan_reservation')
    assert proc.wait(timeout=20) == 75, (root / 'controller-0.log').read_text()
    assert rows(root, 'SELECT state,ref_json FROM cpu_action_reservations') == [('RESERVED', None)]
    assert rows(root, 'SELECT COUNT(*) FROM controller_members') == [(0,)]
    assert ack(root) is None


def test_cpu_closed_actions_cannot_ack_an_unconfirmed_lake_close(cpu_controller):
    root, path, cfg, env, launch, wait_for = cpu_controller
    proc = launch('pass', once=True, fault='close_response_lost')
    assert proc.wait(timeout=20) == 75, (root / 'controller-0.log').read_text()
    assert rows(root, 'SELECT state FROM execution_attempts') == [('TERMINAL',)]
    assert rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',)]
    assert ack(root) is None


def test_cpu_continued_authorization_and_runtime_lease_use_same_stop_proof(cpu_controller):
    root, path, cfg, env, launch, wait_for = cpu_controller
    cfg['execution'].update(version=2, wall_budget_seconds=None)
    cfg['cpu_runtime_lease'] = {'version': 1, 'ttl_seconds': 30}
    path.write_text(yaml.safe_dump(cfg))
    proc = launch('pass', once=True)
    assert proc.wait(timeout=20) == 0, (root / 'controller-0.log').read_text()
    terminal = json.loads(rows(root, 'SELECT terminal_json FROM execution_attempts')[0][0])
    assert terminal['process_tree']['schema'] == 2
    assert ack(root)['resources']['cpu_budget']['reservation_count'] == 1
