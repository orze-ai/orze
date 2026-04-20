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
* ``ORZE_<NAME>_FALLBACK``    — comma-separated list of alternative
  backends to transparently re-dispatch to when the primary hits a
  quota signal (e.g. ``ORZE_CLAUDE_FALLBACK=gemini,codex``). Each
  fallback is invoked as ``orze-<backend>`` with the same argv.

Exit-code contract
------------------
* ``rc == 42`` is the *quota-exhausted sentinel*: emitted when the
  primary (and any in-shim fallback) was specifically blocked by a
  provider quota signal rather than a generic failure. Callers (the
  orze role runner) can observe rc==42 to trigger engine-level
  cross-backend retry.
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
            # Anthropic API 400 message (seen 2026-04 onward):
            "reached your specified api usage limits",
            "api usage limits",
            "rate_limit_error",
            "you will regain access on",
        ),
    ),
    "gemini": BackendSpec(
        binary="gemini",
        api_key_var="GEMINI_API_KEY",
        allow_subscription_mode=False,
        quota_signals=(
            "resource_exhausted",
            "quota exceeded",
            "429 resource has been exhausted",
            "rate limit exceeded",
        ),
    ),
    "codex": BackendSpec(
        binary="codex",
        api_key_var="OPENAI_API_KEY",
        allow_subscription_mode=False,
        quota_signals=(
            "insufficient_quota",
            "rate limit exceeded",
            "you exceeded your current quota",
        ),
    ),
    "kimi": BackendSpec(
        binary="kimi",
        api_key_var="MOONSHOT_API_KEY",
        allow_subscription_mode=False,
        quota_signals=(
            "rate limit exceeded",
            "insufficient_quota",
        ),
    ),
}


# Distinct sentinel return code for "primary (and in-shim retries) blocked
# by a provider quota cap". The role runner uses this to trigger engine-
# level cross-backend fallback.
QUOTA_EXHAUSTED_RC = 42


# Env vars the shim consumes itself — never forward to child.
def _shim_only_vars(backend_name: str) -> Tuple[str, ...]:
    n = backend_name.upper()
    return (
        f"ORZE_{n}_BIN",
        f"ORZE_{n}_FORCE_API",
        f"ORZE_{n}_NO_FALLBACK",
        f"ORZE_{n}_FALLBACK",
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


def _resolve_fallback_backends(backend_name: str, base_env: dict) -> List[str]:
    """Parse ``ORZE_<NAME>_FALLBACK`` into a validated list of backend names.

    When the env var is not set, default to ``gemini`` for ``claude`` backend
    if ``GEMINI_API_KEY`` is available. This matches the documented priority:
    claude subscription -> claude api -> gemini.
    """
    n = backend_name.upper()
    raw = base_env.get(f"ORZE_{n}_FALLBACK", "").strip()
    # Accept truthy boolean shorthand ("1"/"true"/"yes"/"on") as
    # "enable the documented default fallback chain" for this backend.
    if raw.lower() in ("1", "true", "yes", "on"):
        raw = ""  # fall through to default-chain logic below
    if not raw and backend_name == "claude" and base_env.get("GEMINI_API_KEY"):
        raw = "gemini"
    if not raw:
        return []
    out: List[str] = []
    for name in (s.strip().lower() for s in raw.split(",")):
        if not name or name == backend_name:
            continue
        if name not in BACKENDS:
            print(
                f"[orze-{backend_name}] ignoring unknown fallback backend "
                f"{name!r}; known: {', '.join(sorted(BACKENDS))}",
                file=sys.stderr,
            )
            continue
        out.append(name)
    return out


def _dispatch_fallback(fallback_name: str, args: List[str],
                       base_env: dict, primary_name: str) -> int:
    """Re-run as if invoked as ``orze-<fallback_name>``.

    We recurse into ``main()`` with a synthesized argv so the full
    shim contract (subscription retry, quota detection, per-backend
    env hygiene) applies to the fallback too. To prevent infinite
    recursion, we strip ``ORZE_<primary>_FALLBACK`` from the env seen
    by the child dispatch so a bad config can't loop.
    """
    print(
        f"\n[orze-{primary_name}] Cross-backend fallback → {fallback_name}",
        file=sys.stderr,
    )
    os.environ.pop(f"ORZE_{primary_name.upper()}_FALLBACK", None)
    saved_argv0 = sys.argv[0]
    try:
        sys.argv[0] = f"orze-{fallback_name}"
        return main(args)
    finally:
        sys.argv[0] = saved_argv0


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    backend_name, spec = _resolve_backend(sys.argv[0])

    base_env = os.environ.copy()
    n = backend_name.upper()
    claude_bin = base_env.get(f"ORZE_{n}_BIN", spec.binary)
    force_api = base_env.get(f"ORZE_{n}_FORCE_API", "") not in ("", "0")
    no_fallback = base_env.get(f"ORZE_{n}_NO_FALLBACK", "") not in ("", "0")
    api_key_available = bool(base_env.get(spec.api_key_var))
    fallback_backends = (
        [] if no_fallback else _resolve_fallback_backends(backend_name, base_env)
    )

    invocation = [claude_bin, *args]

    # --- Fast paths ---------------------------------------------------
    # Pure passthrough when user opted out entirely.
    if no_fallback:
        child_env = _child_env(
            base_env, spec, backend_name, include_api_key=True,
        )
        try:
            os.execvpe(claude_bin, invocation, child_env)
        except FileNotFoundError:
            print(f"orze-{backend_name}: '{claude_bin}' not found on PATH",
                  file=sys.stderr)
            return 127
        # Unreachable — execvpe replaces the process.

    # No subscription mode OR no key to fall back to → run once; still
    # honor cross-backend fallback on quota-exhaustion.
    if not spec.allow_subscription_mode or not api_key_available:
        rc, captured = _run_once(
            invocation,
            _child_env(base_env, spec, backend_name, include_api_key=True),
        )
        if rc != 0 and _is_quota_exhausted(captured, spec) and fallback_backends:
            next_b, rest = fallback_backends[0], fallback_backends[1:]
            if rest:
                os.environ[f"ORZE_{next_b.upper()}_FALLBACK"] = ",".join(rest)
            return _dispatch_fallback(next_b, args, base_env, backend_name)
        if rc != 0 and _is_quota_exhausted(captured, spec):
            return QUOTA_EXHAUSTED_RC
        return rc

    # Explicit force-API: skip subscription attempt entirely.
    if force_api:
        rc, captured = _run_once(
            invocation,
            _child_env(base_env, spec, backend_name, include_api_key=True),
        )
        if rc != 0 and _is_quota_exhausted(captured, spec) and fallback_backends:
            next_b, rest = fallback_backends[0], fallback_backends[1:]
            if rest:
                os.environ[f"ORZE_{next_b.upper()}_FALLBACK"] = ",".join(rest)
            return _dispatch_fallback(next_b, args, base_env, backend_name)
        if rc != 0 and _is_quota_exhausted(captured, spec):
            return QUOTA_EXHAUSTED_RC
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
        rc, captured = _run_once(
            invocation,
            _child_env(base_env, spec, backend_name, include_api_key=True),
        )
        if rc == 0:
            return 0
        if _is_quota_exhausted(captured, spec) and fallback_backends:
            next_b, rest = fallback_backends[0], fallback_backends[1:]
            if rest:
                os.environ[f"ORZE_{next_b.upper()}_FALLBACK"] = ",".join(rest)
            return _dispatch_fallback(next_b, args, base_env, backend_name)
        if _is_quota_exhausted(captured, spec):
            return QUOTA_EXHAUSTED_RC

    return rc


if __name__ == "__main__":
    sys.exit(main())
