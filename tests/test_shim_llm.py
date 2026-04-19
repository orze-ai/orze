"""Tests for orze.shims.llm — extended quota detection, exit-code sentinel,
cross-backend fallback."""
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from orze.shims import llm


def test_extended_claude_quota_signals_match_anthropic_400():
    spec = llm.BACKENDS["claude"]
    msg = (
        'API Error: 400 {"type":"error","error":'
        '{"type":"invalid_request_error","message":"You have reached '
        'your specified API usage limits. You will regain access on '
        '2026-05-01 at 00:00 UTC."}}'
    )
    assert llm._is_quota_exhausted(msg, spec) is True


def test_gemini_and_codex_registered():
    assert "gemini" in llm.BACKENDS
    assert "codex" in llm.BACKENDS
    assert llm.BACKENDS["gemini"].api_key_var == "GEMINI_API_KEY"


def test_quota_sentinel_constant():
    assert llm.QUOTA_EXHAUSTED_RC == 42


def test_resolve_fallback_backends_parses_and_filters():
    env = {"ORZE_CLAUDE_FALLBACK": "gemini, codex ,  , bogus,claude"}
    fb = llm._resolve_fallback_backends("claude", env)
    assert fb == ["gemini", "codex"]  # drops self, bogus, empty


def test_resolve_fallback_backends_empty_when_unset():
    assert llm._resolve_fallback_backends("claude", {}) == []


def test_main_returns_sentinel_on_quota_without_fallback(monkeypatch):
    """claude shim: force-API path, quota hit, no fallback => rc == 42."""
    monkeypatch.setattr(sys, "argv", ["orze-claude", "-p", "hi"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("ORZE_CLAUDE_FORCE_API", "1")
    monkeypatch.delenv("ORZE_CLAUDE_FALLBACK", raising=False)

    def fake_run_once(argv, env):
        return 1, (
            "API Error: 400 you have reached your specified api usage "
            "limits. you will regain access on 2026-05-01."
        )

    monkeypatch.setattr(llm, "_run_once", fake_run_once)
    rc = llm.main([])
    assert rc == llm.QUOTA_EXHAUSTED_RC


def test_main_dispatches_fallback_on_quota(monkeypatch):
    """ORZE_CLAUDE_FALLBACK=gemini + quota hit => dispatches to gemini shim."""
    monkeypatch.setattr(sys, "argv", ["orze-claude", "-p", "hi"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-fake")
    monkeypatch.setenv("ORZE_CLAUDE_FORCE_API", "1")
    monkeypatch.setenv("ORZE_CLAUDE_FALLBACK", "gemini")

    call_count = {"n": 0}

    def fake_run_once(argv, env):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Primary claude call — quota hit
            return 1, "reached your specified api usage limits"
        # Fallback gemini call — succeed
        return 0, "ok"

    monkeypatch.setattr(llm, "_run_once", fake_run_once)
    rc = llm.main([])
    assert rc == 0
    assert call_count["n"] == 2


def test_main_no_fallback_env_preserves_execvpe_passthrough(monkeypatch):
    """ORZE_CLAUDE_NO_FALLBACK=1 short-circuits to execvpe, even with API key."""
    monkeypatch.setattr(sys, "argv", ["orze-claude", "-p", "hi"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("ORZE_CLAUDE_NO_FALLBACK", "1")

    calls = {}

    def fake_execvpe(bin_, argv, env):
        calls["bin"] = bin_
        raise SystemExit(0)

    monkeypatch.setattr(os, "execvpe", fake_execvpe)
    with pytest.raises(SystemExit):
        llm.main([])
    assert calls["bin"] == "claude"
