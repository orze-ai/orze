from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from orze.core.config import _validate_config
from orze.engine.process import (
    RoleProcess,
    _terminate_and_reap,
    process_descendant_identities,
    process_is_running,
)
from orze.engine.roles import OUTCOME_TIMEOUT, check_active_roles
from orze.engine.tool_policy import (
    build_claude_policy_settings,
    evaluate_tool_call,
    main,
)


def _payload(tool_name: str, tool_input: dict, cwd: Path) -> dict:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "cwd": str(cwd),
    }


def test_config_requires_boolean_policy_switch():
    errors, _ = _validate_config({
        "agent_tool_policy": {"enabled": "sometimes"},
    })

    assert "agent_tool_policy.enabled: must be true or false" in errors


def test_config_rejects_managed_role_policy_bypass():
    errors, _ = _validate_config({
        "agent_tool_policy": {"enabled": True},
        "roles": {
            "engineer": {
                "mode": "claude",
                "skills": ["fixture"],
                "dangerously_skip_permissions": True,
                "claude_args": ["--add-dir", "/"],
            },
        },
    })

    assert any("dangerously_skip_permissions" in error for error in errors)
    assert any("claude_args" in error for error in errors)


def test_file_tools_are_limited_to_workspace_and_resolved_symlinks(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")
    (workspace / "escape").symlink_to(outside)

    assert evaluate_tool_call(
        _payload("Read", {"file_path": "inside.txt"}, workspace),
        [workspace],
    ) is None
    assert evaluate_tool_call(
        _payload("Edit", {"file_path": str(outside)}, workspace),
        [workspace],
    ) == "path_outside_workspace"
    assert evaluate_tool_call(
        _payload("Read", {"file_path": "escape"}, workspace),
        [workspace],
    ) == "path_outside_workspace"


def test_root_scans_are_denied_but_workspace_scans_are_allowed(tmp_path):
    root_scan_commands = [
        "grep -r needle /",
        "rg needle /",
        "find / -name '*.py'",
        "find // -name '*.py'",
        "ugrep -rl token /*",
        "du -sh /",
    ]
    for command in root_scan_commands:
        assert evaluate_tool_call(
            _payload("Bash", {"command": command}, tmp_path), [tmp_path]
        ) == "recursive_root_scan_forbidden"

    assert evaluate_tool_call(
        _payload("Bash", {"command": "rg needle ."}, tmp_path), [tmp_path]
    ) is None


def test_background_and_detached_commands_are_denied(tmp_path):
    cases = [
        {"command": "python worker.py", "run_in_background": True},
        {"command": "python worker.py &"},
        {"command": "nohup python worker.py"},
        {"command": "setsid python worker.py"},
        {"command": "tmux new-session -d worker"},
        {"command": "docker run -d image"},
    ]
    for tool_input in cases:
        assert evaluate_tool_call(
            _payload("Bash", tool_input, tmp_path), [tmp_path]
        ) == "detached_process_forbidden"


def test_denial_audit_hashes_but_never_stores_command(tmp_path, monkeypatch):
    secret = "credential-that-must-not-be-persisted"
    audit_log = tmp_path / "policy.jsonl"
    payload = _payload(
        "Bash", {"command": f"grep -r {secret} /"}, tmp_path)
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)

    assert main([
        "--allow-root", str(tmp_path), "--audit-log", str(audit_log),
    ]) == 0

    response = json.loads(stdout.getvalue())
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"
    audit_text = audit_log.read_text(encoding="utf-8")
    audit = json.loads(audit_text)
    assert audit["reason"] == "recursive_root_scan_forbidden"
    assert len(audit["input_sha256"]) == 64
    assert secret not in audit_text


def test_generated_settings_fail_closed_and_reallow_only_runtime_roots(tmp_path):
    project = tmp_path / "project"
    settings = build_claude_policy_settings(
        project, project / ".orze" / "logs" / "policy.jsonl")

    sandbox = settings["sandbox"]
    assert sandbox["enabled"] is True
    assert sandbox["failIfUnavailable"] is True
    assert sandbox["allowUnsandboxedCommands"] is False
    assert str(project.resolve()) in sandbox["filesystem"]["allowRead"]
    assert sandbox["filesystem"]["allowWrite"] == [str(project.resolve())]
    hook = settings["hooks"]["PreToolUse"][0]
    assert hook["matcher"] == "Bash|Read|Write|Edit|Glob|Grep"
    assert "tool_policy.py" in hook["hooks"][0]["command"]


def test_generated_hook_command_denies_without_pythonpath(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    audit_log = project / ".orze" / "logs" / "policy.jsonl"
    settings = build_claude_policy_settings(project, audit_log)
    hook_cmd = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    payload = _payload("Bash", {"command": "find / -name token"}, project)

    result = subprocess.run(
        shlex.split(hook_cmd), input=json.dumps(payload), text=True,
        capture_output=True, env={"PATH": os.environ.get("PATH", "")},
        timeout=5,
    )

    assert result.returncode == 0
    response = json.loads(result.stdout)
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert json.loads(audit_log.read_text(encoding="utf-8"))["reason"] == (
        "recursive_root_scan_forbidden")


def test_reaper_kills_descendant_that_escaped_process_group(tmp_path):
    child_pid_file = tmp_path / "escaped.pid"
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)'], start_new_session=True)\n"
        "Path(os.environ['CHILD_PID_FILE']).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["CHILD_PID_FILE"] = str(child_pid_file)
    parent = subprocess.Popen(
        [sys.executable, str(parent_script)], env=env, start_new_session=True)
    deadline = time.time() + 3
    while not child_pid_file.exists() and time.time() < deadline:
        time.sleep(0.01)
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    descendants = process_descendant_identities(parent.pid)
    assert any(item["pid"] == child_pid for item in descendants)
    assert any(item["pgid"] != parent.pid for item in descendants)

    _terminate_and_reap(parent, "escaped fixture", timeout=0.2,
                        pgid=parent.pid)

    assert parent.poll() is not None
    assert not process_is_running(child_pid)


def test_role_timeout_reaps_escaped_descendant(tmp_path):
    child_pid_file = tmp_path / "role-child.pid"
    parent_script = tmp_path / "role-parent.py"
    parent_script.write_text(
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)'], start_new_session=True)\n"
        "Path(os.environ['CHILD_PID_FILE']).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["CHILD_PID_FILE"] = str(child_pid_file)
    log_path = tmp_path / "role.log"
    log_fh = log_path.open("w", encoding="utf-8")
    parent = subprocess.Popen(
        [sys.executable, str(parent_script)], env=env,
        stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + 3
    while not child_pid_file.exists() and time.time() < deadline:
        time.sleep(0.01)
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    lock_dir = tmp_path / "role.lock"
    lock_dir.mkdir()
    active = {
        "engineer": RoleProcess(
            role_name="engineer", process=parent,
            start_time=time.time() - 10, log_path=log_path, timeout=1,
            lock_dir=lock_dir, cycle_num=1, _log_fh=log_fh,
            writes_ideas_file=False,
        ),
    }

    finished = check_active_roles(active, ideas_file=str(tmp_path / "ideas.md"))

    assert finished == [("engineer", OUTCOME_TIMEOUT)]
    assert active == {}
    assert parent.poll() is not None
    assert not process_is_running(child_pid)
