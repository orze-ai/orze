"""orze-claude — a drop-in wrapper for `claude -p` that falls back from
a Claude Code subscription to the direct Anthropic API when the
subscription hits its extra-usage quota.

Motivation
----------
When an orze pipeline runs on a host signed into a Claude Code
subscription, every `mode: claude` role cycle (professor, engineer,
data_analyst, thinker, code_evolution) consumes subscription extra
usage. Busy pipelines exhaust the daily cap — often in the middle of
the afternoon — and every role cycle after that exits non-zero with:

    "You're out of extra usage · resets 7pm (UTC)"

If the user *also* has an `ANTHROPIC_API_KEY`, they pay twice in the
worst case (subscription + API) or they give up one or the other. This
shim lets them keep both: **subscription is tried first (free within
the cap); the API key covers the overflow automatically.**

Behaviour
---------
1. If ``ANTHROPIC_API_KEY`` is *not* in the environment, this is a pure
   passthrough to ``claude`` — no retry logic, no overhead.
2. If ``ANTHROPIC_API_KEY`` is set:
   a. First attempt: env with the key stripped → subscription path.
   b. Stream the child's merged stdout+stderr to our stdout line by
      line so orze's role log captures it in real time.
   c. On exit, if returncode != 0 AND the captured output matches a
      known quota signal, re-launch with ``ANTHROPIC_API_KEY`` in env
      (API path) and stream that run too.
   d. Exit with the final subprocess's return code.

Escape hatches
--------------
* ``ORZE_CLAUDE_BIN`` — which underlying binary to call (default
  ``claude``). Useful for testing or when multiple CLIs coexist.
* ``ORZE_CLAUDE_FORCE_API=1`` — skip the subscription attempt, go
  straight to the API key. Handy once the user knows their
  subscription is exhausted and wants to avoid the wasted first call.
* ``ORZE_CLAUDE_NO_FALLBACK=1`` — disable the retry entirely. Behaves
  exactly like bare ``claude``.

Exit codes follow the child's exit codes verbatim.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Tuple


# Quota-exhaustion strings observed in Claude Code CLI stdout/stderr.
# Kept conservative: we match only phrases that unambiguously indicate
# the subscription cap was hit, so we never retry on a different error.
_QUOTA_SIGNALS: Tuple[str, ...] = (
    "out of extra usage",
    "usage limit reached",
    "monthly limit reached",
    "subscription limit",
)

# Environment variables the shim itself consumes (for configuration).
# We never forward these to the child — they're ours.
_SHIM_ONLY_VARS = ("ORZE_CLAUDE_BIN",
                   "ORZE_CLAUDE_FORCE_API",
                   "ORZE_CLAUDE_NO_FALLBACK")


def _child_env(base_env: dict, include_api_key: bool) -> dict:
    """Return a child env with ANTHROPIC_API_KEY present or stripped.

    We always strip the shim's own config vars — they have no meaning
    to the underlying claude CLI and could confuse future changes.
    """
    env = {k: v for k, v in base_env.items()
           if k not in _SHIM_ONLY_VARS and (
               include_api_key or k != "ANTHROPIC_API_KEY")}
    return env


def _run_once(argv: List[str], env: dict) -> Tuple[int, str]:
    """Launch argv with env, stream merged output to our stdout, return
    (returncode, captured_output_string).
    """
    # Merge stderr into stdout so the quota check sees messages wherever
    # the CLI wrote them. Line-buffered so orze's role log captures
    # progress in real time (claude writes tool-use events incrementally).
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


def _is_quota_exhausted(output: str) -> bool:
    low = output.lower()
    return any(sig in low for sig in _QUOTA_SIGNALS)


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    base_env = os.environ.copy()
    claude_bin = base_env.get("ORZE_CLAUDE_BIN", "claude")
    force_api = base_env.get("ORZE_CLAUDE_FORCE_API", "") not in ("", "0")
    no_fallback = base_env.get("ORZE_CLAUDE_NO_FALLBACK", "") not in ("", "0")
    api_key_available = bool(base_env.get("ANTHROPIC_API_KEY"))

    invocation = [claude_bin, *args]

    # Fast path: no API key, or user explicitly disabled the shim.
    # Passthrough with a single attempt — no output capture, no retry.
    if no_fallback or not api_key_available:
        # If no API key and the shim is in charge, there's nothing to
        # retry anyway — just exec-replace ourselves with claude.
        try:
            os.execvpe(claude_bin, invocation, _child_env(base_env, False))
        except FileNotFoundError:
            print(f"orze-claude: '{claude_bin}' not found on PATH",
                  file=sys.stderr)
            return 127

    # Explicit force-API path: skip subscription attempt.
    if force_api:
        rc, _ = _run_once(invocation, _child_env(base_env, include_api_key=True))
        return rc

    # Default path: try subscription first, fall back to API on quota.
    rc, captured = _run_once(invocation,
                             _child_env(base_env, include_api_key=False))
    if rc == 0:
        return 0

    if _is_quota_exhausted(captured):
        print("\n[orze-claude] Subscription quota hit — retrying with "
              "ANTHROPIC_API_KEY.", file=sys.stderr)
        rc, _ = _run_once(invocation,
                          _child_env(base_env, include_api_key=True))
        return rc

    # Non-quota failure (bad prompt, auth revoked, whatever) — don't
    # retry, let orze surface the error as normal.
    return rc


if __name__ == "__main__":
    sys.exit(main())
