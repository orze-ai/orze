"""Registered input/decode failures must not become optional GPU inventory.

This uses an explicit context spy; it checks the adapter exception boundary,
not persistent registration. The old no-context subprocess contract is kept.
"""
import pytest

from orze.engine import controller_probe as probe
from test_controller_probe import owned, _run


@pytest.mark.parametrize("options", [{"shell": True}, {"errors": "ignore"}, {"errors": "replace"}])
def test_unsupported_or_lossy_registered_probe_input_is_sticky_hold_before_prepare(owned, options):
    with pytest.raises(probe.ControllerProbeHOLD, match="input_invalid"):
        _run(owned, "import os; os.write(1,b'\\xff')", **options)
    assert owned.handles == [] and owned.settled == []
    assert owned.holds == ["controller_probe_input_invalid"]
