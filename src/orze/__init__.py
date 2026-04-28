"""orze — orze.ai."""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("orze")
except PackageNotFoundError:
    __version__ = "unknown"

from orze.journal import Journal, Iteration

__all__ = ["Journal", "Iteration"]
