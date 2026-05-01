"""Round-2 A1: shim auth-failure fallback fires by default (no env var).

Verifies the cache helpers and the early-kill predicate. The full
subprocess path is exercised with a fake ``claude`` binary that prints
'Not logged in' to stderr and exits.
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orze.shims import llm  # noqa: E402


def test_auth_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv(llm._AUTH_CACHE_DIR_ENV, str(tmp_path))
    assert llm._read_auth_mode_cache("claude") is None
    llm._write_auth_mode_cache("claude", "api")
    assert llm._read_auth_mode_cache("claude") == "api"
    llm._write_auth_mode_cache("claude", "subscription")
    assert llm._read_auth_mode_cache("claude") == "subscription"
    # Bogus values are ignored.
    llm._write_auth_mode_cache("claude", "junk")
    assert llm._read_auth_mode_cache("claude") == "subscription"


def test_auth_failure_predicate():
    spec = llm.BACKENDS["claude"]
    assert llm._is_auth_failure("Not logged in · Please run /login", spec)
    assert llm._is_auth_failure("authentication failed: foo", spec)
    assert not llm._is_auth_failure("hello world", spec)


def test_early_kill_aborts_on_auth_failure(tmp_path, monkeypatch):
    """End-to-end smoke: a fake binary that prints the auth-failure
    string within 1 second of startup should be killed by
    _run_with_early_auth_kill."""
    fake = tmp_path / "fake-claude"
    fake.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import sys, time
        sys.stdout.write("Not logged in - Please run /login\\n")
        sys.stdout.flush()
        time.sleep(60)  # would hang forever — early-kill must abort us
    """))
    fake.chmod(0o755)
    spec = llm.BACKENDS["claude"]
    rc, captured, killed = llm._run_with_early_auth_kill(
        [str(fake)], dict(os.environ), spec, "claude")
    assert killed is True, f"expected killed=True; captured={captured!r} rc={rc}"
    assert "not logged in" in captured.lower()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        os.environ[llm._AUTH_CACHE_DIR_ENV] = td
        try:
            test_auth_failure_predicate()
            from pathlib import Path as _P
            tmp = _P(td)
            os.environ[llm._AUTH_CACHE_DIR_ENV] = str(tmp)
            llm._write_auth_mode_cache("claude", "api")
            assert llm._read_auth_mode_cache("claude") == "api"
            print("test_auth_cache_roundtrip OK")
            print("test_auth_failure_predicate OK")
            test_early_kill_aborts_on_auth_failure(tmp, type("_M", (), {"setenv": lambda *a: None})())
            print("test_early_kill_aborts_on_auth_failure OK")
        finally:
            os.environ.pop(llm._AUTH_CACHE_DIR_ENV, None)
