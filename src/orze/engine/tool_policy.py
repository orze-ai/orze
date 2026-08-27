"""Fail-closed policy hook for autonomous agent tool calls.

The hook is intentionally deterministic. It blocks workspace escapes through
built-in file tools, recursive scans rooted at ``/``, and common ways for Bash
commands to detach from the role process. Denials are audited by hash only so a
command containing credentials is never copied into Orze metadata.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Iterable, Optional


_FILE_PATH_FIELDS = {
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "Glob": ("path",),
    "Grep": ("path",),
}
_DETACH_COMMANDS = {
    "at", "batch", "daemon", "disown", "nohup", "screen", "setsid",
    "systemd-run", "tmux",
}
_ROOT_SCAN_COMMANDS = {
    "du", "fd", "fdfind", "find", "grep", "egrep", "fgrep", "rg",
    "ripgrep", "ugrep",
}


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    for root in roots:
        try:
            resolved.relative_to(root.expanduser().resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def _tokens(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        # An unparseable shell command must not bypass the policy. Returning a
        # sentinel lets evaluate_tool_call deny it without storing the input.
        return ["<unparseable>"]


def _is_root_scan(tokens: list[str]) -> bool:
    commands = {Path(token).name for token in tokens}
    root_target = False
    for token in tokens:
        if token.startswith("/*"):
            root_target = True
            break
        if token.startswith("/"):
            try:
                if Path(token).resolve(strict=False) == Path("/"):
                    root_target = True
                    break
            except (OSError, RuntimeError):
                continue
    return bool(commands & _ROOT_SCAN_COMMANDS) and root_target


def _is_detached(tokens: list[str], tool_input: dict) -> bool:
    if tool_input.get("run_in_background") is True:
        return True
    if "&" in tokens:
        return True
    commands = {Path(token).name for token in tokens}
    if commands & _DETACH_COMMANDS:
        return True
    for runtime in ("docker", "podman"):
        if runtime in commands and "run" in tokens and "-d" in tokens:
            return True
    return False


def evaluate_tool_call(payload: dict, allowed_roots: Iterable[Path]
                       ) -> Optional[str]:
    """Return a stable denial reason, or ``None`` when the call is allowed."""
    roots = [Path(root).resolve(strict=False) for root in allowed_roots]
    if not roots:
        return "workspace_roots_missing"
    if not isinstance(payload, dict):
        return "invalid_hook_payload"
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if not isinstance(tool_name, str) or not isinstance(tool_input, dict):
        return "invalid_hook_payload"

    cwd = Path(payload.get("cwd") or roots[0]).resolve(strict=False)
    for field in _FILE_PATH_FIELDS.get(tool_name, ()):
        raw = tool_input.get(field)
        if raw in (None, ""):
            continue
        if not isinstance(raw, str):
            return "invalid_tool_path"
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if not _inside(candidate, roots):
            return "path_outside_workspace"

    if tool_name == "Bash":
        command = tool_input.get("command")
        if not isinstance(command, str) or not command.strip():
            return "invalid_bash_command"
        tokens = _tokens(command)
        if tokens == ["<unparseable>"]:
            return "unparseable_bash_command"
        if _is_detached(tokens, tool_input):
            return "detached_process_forbidden"
        if _is_root_scan(tokens):
            return "recursive_root_scan_forbidden"
    return None


def _audit(log_path: Optional[Path], payload: object, reason: str) -> None:
    if log_path is None:
        return
    try:
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        record = {
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "tool_name": (
                payload.get("tool_name", "unknown")
                if isinstance(payload, dict) else "unknown"
            ),
            "reason": reason,
            "input_sha256": hashlib.sha256(canonical).hexdigest(),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(record, separators=(",", ":")) + "\n")
                     .encode("utf-8"))
        finally:
            os.close(fd)
    except (OSError, TypeError, ValueError):
        # Policy enforcement must not become permissive when auditing fails.
        pass


def build_claude_policy_settings(project_root: Path, audit_log: Path,
                                 python: Optional[str] = None) -> dict:
    """Build strict Claude Code settings for an Orze-managed role."""
    project_root = Path(project_root).resolve(strict=False)
    audit_log = Path(audit_log).resolve(strict=False)
    if project_root.parent == project_root:
        raise ValueError("agent tool policy cannot authorize filesystem root")
    if not _inside(audit_log, [project_root]):
        raise ValueError("agent tool policy audit log must stay in project")
    # Invoke this exact reviewed file with the base interpreter. ``python -m``
    # from writable role scratch would allow a shadow ``orze`` package to
    # replace the policy; a project-local venv interpreter is writable by the
    # engineering role for the same reason.
    policy_script = Path(__file__).resolve()
    policy_python = (
        python or getattr(sys, "_base_executable", None) or sys.executable)
    hook_cmd = shlex.join([
        str(policy_python), str(policy_script),
        "--allow-root", str(project_root),
        "--audit-log", str(audit_log),
    ])

    # Deny the broad data/user mounts that made an accidental `grep -r /`
    # expensive and reputation-sensitive. The project root is explicitly
    # re-allowed; runtime library roots are read-only exceptions so tests and
    # linters can execute inside the OS sandbox.
    deny_roots = [
        str(Path.home().resolve(strict=False)),
        "/ceph", "/data", "/hot-data", "/mnt", "/root", "/workspace",
        "/proc", "/sys",
    ]
    runtime_roots = {
        str(project_root), str(Path(sys.prefix).resolve()),
        "/dev", "/etc/hosts", "/etc/resolv.conf", "/etc/ssl",
        "/proc/self", "/proc/thread-self", "/tmp",
    }
    for entry in sys.path:
        if entry:
            path = Path(entry)
            if path.is_absolute() and path.exists():
                runtime_roots.add(str(path.resolve()))

    return {
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                "denyRead": sorted(set(deny_roots)),
                "allowRead": sorted(runtime_roots),
                "allowWrite": [str(project_root)],
            },
        },
        "hooks": {
            "PreToolUse": [{
                "matcher": "Bash|Read|Write|Edit|Glob|Grep",
                "hooks": [{
                    "type": "command",
                    "command": hook_cmd,
                    "timeout": 5,
                }],
            }],
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-root", action="append", required=True)
    parser.add_argument("--audit-log")
    args = parser.parse_args(argv)
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        payload = None
    reason = evaluate_tool_call(
        payload, [Path(root) for root in args.allow_root])
    if reason is None:
        return 0
    _audit(Path(args.audit_log) if args.audit_log else None, payload, reason)
    json.dump({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
