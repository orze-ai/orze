"""Observe the SQLite allocator in this isolated measurement process only.

Definitions: https://www.sqlite.org/c3ref/status.html and
https://www.sqlite.org/c3ref/c_status_malloc_count.html . MEMORY_USED excludes
separately configured auxiliary page-cache memory; it is not Python heap or RSS.
No product module imports this optional measurement helper.
"""
import ctypes
import sqlite3
import _sqlite3

_library = ctypes.CDLL(_sqlite3.__file__)
_library.sqlite3_libversion.restype = ctypes.c_char_p
assert _library.sqlite3_libversion().decode() == sqlite3.sqlite_version
_status = _library.sqlite3_status64
_status.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.c_int]
_status.restype = ctypes.c_int


def snapshot(*, reset=False):
    current, peak = ctypes.c_longlong(), ctypes.c_longlong()
    assert _status(0, ctypes.byref(current), ctypes.byref(peak), int(reset)) == 0
    return {"current": current.value, "peak": peak.value}
