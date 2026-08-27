"""Subprocess wrapper that activates data boundary guardrails before user code.

CALLING SPEC:
    python -m orze.data_boundaries.wrap <script.py> [args...]
        1. Call orze.data_boundaries.activate() — reads env vars, patches open()
        2. Exec <script.py> as __main__ with the remaining argv.

Used by orze.engine.launcher when data_boundaries is configured.
The wrapper exists so that user training scripts need zero modification
to benefit from the guardrail: they just get launched via the wrapper
instead of directly.
"""
import os
import runpy
import sys

from orze.data_boundaries import activate, is_active


def _attest_kernel_boundary() -> None:
    """Acknowledge isolation through a parent-created one-shot pipe."""
    fd_text = os.environ.pop("ORZE_BOUNDARY_ATTEST_FD", None)
    nonce = os.environ.pop("ORZE_BOUNDARY_ATTEST_NONCE", None)
    if fd_text is None and nonce is None:
        return
    fd = None
    try:
        if (fd_text is None or nonce is None or len(nonce) != 64
                or any(char not in "0123456789abcdef" for char in nonce)
                or os.environ.get("ORZE_KERNEL_BOUNDARY_ACTIVE") != "1"):
            raise ValueError("invalid attestation environment")
        fd = int(fd_text)
        if fd < 3:
            raise ValueError("invalid attestation fd")
        encoded = (nonce + "\n").encode("ascii")
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short attestation write")
            view = view[written:]
    except (OSError, TypeError, ValueError):
        sys.stderr.write(
            "[orze data_boundaries] boundary attestation failed\n")
        sys.exit(126)
    finally:
        if isinstance(fd, int) and fd >= 3:
            try:
                os.close(fd)
            except OSError:
                pass


def main() -> None:
    if (os.environ.get("ORZE_REQUIRE_KERNEL_BOUNDARY") == "1"
            and os.environ.get("ORZE_KERNEL_BOUNDARY_ACTIVE") != "1"):
        # Never execute user code if a future launcher refactor drops the
        # mandatory namespace wrapper. Keep the error content-free.
        sys.stderr.write(
            "[orze data_boundaries] required kernel boundary is inactive\n"
        )
        sys.exit(126)
    _attest_kernel_boundary()
    activate()  # no-op if ORZE_FORBIDDEN_PATHS / ORZE_WATCH_PATHS unset

    if len(sys.argv) < 2:
        sys.stderr.write("usage: python -m orze.data_boundaries.wrap <script.py> [args...]\n")
        sys.exit(2)

    script = sys.argv[1]
    # Shift argv so the user script sees a clean sys.argv starting at [script, ...]
    sys.argv = sys.argv[1:]

    if is_active():
        sys.stderr.write(f"[orze data_boundaries] guardrails active\n")

    # runpy preserves __name__ == '__main__' semantics
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
