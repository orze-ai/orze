"""Star prompt — shown once on first run.

Calling spec:
    from orze.cli_star import maybe_star
    maybe_star()  # idempotent, writes ~/.orze_starred marker
"""

import shutil
import subprocess
from pathlib import Path

_STAR_REPO = "erikhenriksson/orze"
_STAR_MARKER = Path.home() / ".orze_starred"


def maybe_star():
    """Prompt user to star the repo on first run (once only)."""
    if _STAR_MARKER.exists():
        return
    if not shutil.which("gh"):
        _STAR_MARKER.touch()
        return
    # Check if gh is authenticated
    try:
        subprocess.run(
            ["gh", "auth", "status"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        _STAR_MARKER.touch()
        return
    try:
        answer = input(
            f"\n\033[1mEnjoy Orze?\033[0m Press Enter to \u2b50 us on GitHub, "
            f"or type N to skip: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        _STAR_MARKER.touch()
        return
    if answer in ("", "y", "yes"):
        try:
            subprocess.run(
                ["gh", "api", "-X", "PUT", f"/user/starred/{_STAR_REPO}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            print("\033[32mThanks for starring!\033[0m\n")
        except Exception:
            pass
    _STAR_MARKER.touch()
