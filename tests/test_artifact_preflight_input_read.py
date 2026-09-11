"""New nonblocking read mechanism; never run the old blocking FIFO path."""
import os

import pytest

from orze.engine import artifact_preflight_receipts as receipts
from orze.engine.termination_hold import TerminationUnconfirmed
from test_artifact_preflight_source import resolver, _passed
from test_native_artifact_preflight_tree_completion import cpu_preflight
from test_native_pre_script_tree_completion import cpu_pre_script


def test_regular_input_replaced_by_fifo_after_stat_fails_without_open_block(resolver, monkeypatch):
    c = resolver
    _passed(c)
    actual = receipts._files
    observed = []

    def replace_after_stat(folder, cfg):
        value = actual(folder, cfg)
        if not observed:
            observed.append(value)
            c.resolver.unlink()
            os.mkfifo(c.resolver)
        return value

    monkeypatch.setattr(receipts, "_files", replace_after_stat)
    with pytest.raises(TerminationUnconfirmed):
        receipts.capture_preflight_source(c.lake, c.folder, c.cfg)
    assert len(observed) == 1
    assert not (c.folder / "_compute_receipts").exists()
