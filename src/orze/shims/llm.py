"""Drop-in wrappers for LLM CLIs that transparently handle subscription-
vs-API-key authentication and quota-exhaustion fallback.

Design
------
One Python function, N console-script entry points. Each entry point
corresponds to a supported LLM CLI — its name is the dispatch key:

    orze-claude   -> wraps `claude` (subscription-first, API-key fallback)
    orze-codex    -> (future) wraps `codex`
    orze-gemini   -> (future) wraps `gemini`
    orze-kimi     -> (future) wraps `kimi`

At runtime `main()` reads ``sys.argv[0]``, strips the ``orze-`` prefix,
and looks up the backend spec in ``BACKENDS``. Everything else — the
retry policy, env shaping, streaming subprocess runner — is shared.

Why separate entry points rather than a generic ``orze-llm claude …``:
**drop-in replacement**. Every existing script, Makefile, and editor
integration that calls ``claude`` keeps working by changing one token.
A generic dispatcher would force every caller to re-learn argv.

Why all entries route to the same function: adding a new backend is an
entry in ``BACKENDS`` + a line in ``pyproject.toml`` — not a new file.

Backend contract
----------------
A backend spec is a ``BackendSpec`` dataclass:

* ``binary`` — the underlying CLI to exec (overridable via
  ``ORZE_<NAME>_BIN``, e.g. ``ORZE_CLAUDE_BIN``).
* ``api_key_var`` — env var holding the API key. If set and
  ``allow_subscription_mode=True``, the first attempt strips it to
  force the subscription path; the shim falls back to the key when the
  output matches a quota signal.
* ``allow_subscription_mode`` — ``True`` for CLIs that support a
  browser/OAuth subscription alongside the env-var key (notably
  Claude Code). ``False`` for API-key-only CLIs (typical OpenAI /
  Gemini / Moonshot tools), where the fallback semantics don't apply
  — those still get streaming output and the shared PID/env hygiene.
* ``quota_signals`` — substrings (case-insensitive) that unambiguously
  indicate the subscription/quota cap was hit. **Must be empirically
  verified** against real quota-exhausted output for that CLI before
  enabling ``allow_subscription_mode``.

Escape hatches (per backend, env-var prefix is the uppercased backend
name — e.g. ``CLAUDE``):

* ``ORZE_<NAME>_BIN``         — override the underlying binary
* ``ORZE_<NAME>_FORCE_API``   — skip subscription attempt, use key
* ``ORZE_<NAME>_NO_FALLBACK`` — pure passthrough, no retry logic
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class BackendSpec:
    """How to wrap one LLM CLI. See module docstring for field semantics."""

    binary: str
    api_key_var: str
    allow_subscription_mode: bool
    quota_signals: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Backend registry. Add a new LLM CLI by extending this dict AND adding an
# `orze-<name>` console-script entry in pyproject.toml pointing at
# `orze.shims.llm:main`. No Python changes otherwise.
# ---------------------------------------------------------------------------
BACKENDS: dict = {
    "claude": BackendSpec(
        binary="claude",
        api_key_var="ANTHROPIC_API_KEY",
        allow_subscription_mode=True,
        quota_signals=(
            "out of extra usage",
            "usage limit reached",
            "monthly limit reached",
            "subscription limit",
        ),
    ),
    # Future — unverified quota strings, add once we've seen them in the wild:
    # "codex": BackendSpec(
    #     binary="codex", api_key_var="OPENAI_API_KEY",
    #     allow_subscription_mode=False, quota_signals=(),
    # ),
    # "gemini": BackendSpec(
    #     binary="gemini", api_key_var="GEMINI_API_KEY",
    #     allow_subscription_mode=False, quota_signals=(),
    # ),
    # "kimi": BackendSpec(
    #     binary="kimi", api_key_var="MOONSHOT_API_KEY",
    #     allow_subscription_mode=False, quota_signals=(),
    # ),
}


# Env vars the shim consumes itself — never forward to child.
def _shim_only_vars(backend_name: str) -> Tuple[str, ...]:
    n = backend_name.upper()
    return (
        f"ORZE_{n}_BIN",
        f"ORZE_{n}_FORCE_API",
        f"ORZE_{n}_NO_FALLBACK",
    )


def _child_env(base_env: dict, spec: BackendSpec, backend_name: str,
               include_api_key: bool) -> dict:
    """Return a child env with the API key optionally stripped, and the
    shim's own config vars always removed."""
    shim_vars = set(_shim_only_vars(backend_name))
    return {
        k: v for k, v in base_env.items()
        if k not in shim_vars and (include_api_key or k != spec.api_key_var)
    }


def _run_once(argv: List[str], env: dict) -> Tuple[int, str]:
    """Launch argv with env; stream merged stdout+stderr to our stdout line
    by line; return (returncode, captured_output).

    Streaming matters — orze's role log needs to see the child's
    progress in real time, not all-at-once at exit.
    """
    proc = subprocess.Popen(
        argv, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    assert proc.stdout is not None
    captured: List[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()
    return proc.returncode, "".join(captured)


def _is_quota_exhausted(output: str, spec: BackendSpec) -> bool:
    if not spec.quota_signals:
        return False
    low = output.lower()
    return any(sig in low for sig in spec.quota_signals)


def _resolve_backend(argv0: str) -> Tuple[str, BackendSpec]:
    """Dispatch on the console-script basename: `orze-claude` -> `claude`."""
    name = Path(argv0).name
    if name.startswith("orze-"):
        name = name[len("orze-"):]
    spec = BACKENDS.get(name)
    if spec is None:
        known = ", ".join(sorted(BACKENDS))
        print(
            f"orze-shim: unknown backend {name!r}; expected one of: {known}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return name, spec


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    backend_name, spec = _resolve_backend(sys.argv[0])

    base_env = os.environ.copy()
    n = backend_name.upper()
    claude_bin = base_env.get(f"ORZE_{n}_BIN", spec.binary)
    force_api = base_env.get(f"ORZE_{n}_FORCE_API", "") not in ("", "0")
    no_fallback = base_env.get(f"ORZE_{n}_NO_FALLBACK", "") not in ("", "0")
    api_key_available = bool(base_env.get(spec.api_key_var))

    invocation = [claude_bin, *args]

    # --- Fast paths ---------------------------------------------------
    # Pure passthrough when nothing to fall back to, or user opted out.
    # Exec-replace to avoid one extra Python process hanging around.
    if no_fallback or not spec.allow_subscription_mode or not api_key_available:
        child_env = _child_env(
            base_env, spec, backend_name,
            # For API-key-only CLIs, the key must be present. For claude
            # with no key set, there's nothing to strip anyway.
            include_api_key=True,
        )
        try:
            os.execvpe(claude_bin, invocation, child_env)
        except FileNotFoundError:
            print(f"orze-{backend_name}: '{claude_bin}' not found on PATH",
                  file=sys.stderr)
            return 127
        # Unreachable — execvpe replaces the process.

    # Explicit force-API: skip subscription attempt entirely.
    if force_api:
        rc, _ = _run_once(invocation,
                          _child_env(base_env, spec, backend_name,
                                     include_api_key=True))
        return rc

    # --- Default path: subscription first, API-key fallback on quota signal.
    rc, captured = _run_once(
        invocation,
        _child_env(base_env, spec, backend_name, include_api_key=False),
    )
    if rc == 0:
        return 0

    if _is_quota_exhausted(captured, spec):
        print(
            f"\n[orze-{backend_name}] Subscription quota hit — retrying "
            f"with {spec.api_key_var}.",
            file=sys.stderr,
        )
        rc, _ = _run_once(
            invocation,
            _child_env(base_env, spec, backend_name, include_api_key=True),
        )
        return rc

    return rc


if __name__ == "__main__":
    sys.exit(main())
