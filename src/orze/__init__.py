"""orze — orze.ai."""
import re
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _source_version():
    """Return the checkout version when importing directly from ``src/``.

    Package metadata can describe a different installed Orze release while
    ``PYTHONPATH`` points at a source checkout.  Prefer that checkout's
    pyproject version so upgrade comparisons never downgrade live code.
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


__version__ = _source_version()
if __version__ is None:
    try:
        __version__ = _pkg_version("orze")
    except PackageNotFoundError:
        __version__ = "unknown"

from orze.journal import Journal, Iteration

__all__ = ["Journal", "Iteration"]
