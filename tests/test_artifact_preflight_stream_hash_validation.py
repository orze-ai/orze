"""Bounded decoded-receipt validation, not a file-SHA forgery experiment.

The explicit decoder seam retains the verified file digest while replacing one
decoded field. It tests required field validation independently of hashing; the
separate receipt-replacement case exercises actual bytes/SHA rejection.
"""
from copy import deepcopy

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import native_artifact_preflight as native
from test_artifact_preflight_supervision_binding import resolver


@pytest.mark.parametrize("fault", ["missing_stdout", "invalid_stderr"])
def test_cached_executed_result_requires_two_measured_sha256_fields(resolver, monkeypatch, fault):
    a = resolver
    a.run()
    row = current_attempt(a.lake.conn, a.idea, "artifact_preflight")
    original = native._read
    path = a.folder / "artifact_preflight.json"

    def read(candidate, *args, **kwargs):
        value, digest = original(candidate, *args, **kwargs)
        if candidate == path:
            value = deepcopy(value)
            if fault == "missing_stdout":
                value.pop("stdout_sha256")
            else:
                value["stderr_sha256"] = "not-a-measured-sha256"
        return value, digest

    monkeypatch.setattr(native, "_read", read)
    with pytest.raises(native.ArtifactPreflightHOLD):
        native._cached(row, a.folder)
    assert len(a.prepared) == 1
