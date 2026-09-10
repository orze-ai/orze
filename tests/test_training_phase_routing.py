"""Draft routing regressions, using an actual supervised native CPU tree.

A mutable posthoc flag must not redirect native training to a legacy STOP or
publication path. These cases prove misrouted stop effects, not early artifact
acceptance: the supervised handle itself still prevents a live-tree integer.
"""
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_training_tree_completion import cpu_training, native_case, _launch, _alive


@pytest.mark.parametrize("keep_ref", [True, False], ids=["native-ref", "tokenless-native-history"])
def test_posthoc_flag_cannot_redirect_native_training_stop(cpu_training, tmp_path, keep_ref):
    c = cpu_training
    tp = _launch(c, tmp_path, detached=True)
    assert tp.process.poll() is None
    assert _alive(c.daemon_pidfd)
    before = current_attempt(c.lake.conn, c.idea, "training")
    metrics = (c.folder / "metrics.json").read_bytes()
    claim = (c.folder / "claim.json").read_bytes()
    assert not (c.folder / "_execution_stops").exists()
    tp.is_posthoc = True
    tp.timeout = 0
    if not keep_ref:
        tp.attempt_ref = None
    active, failures = {0: tp}, {}
    try:
        launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake)
    except (AttemptEffectBusy, AttemptEffectInDoubt, TerminationUnconfirmed):
        pass
    requests = list(c.folder.glob("_execution_stops/*/requested.json"))
    assert requests == [], (
        "mutable phase metadata authorized a real wrongly-labelled STOP",
        [json.loads(path.read_bytes())["phase"] for path in requests])
    assert _alive(c.daemon_pidfd), "contradictory phase cannot authorize stopping the owned tree"
    assert tp.process.poll() is None
    assert current_attempt(c.lake.conn, c.idea, "training") == before
    assert (c.folder / "metrics.json").read_bytes() == metrics
    assert (c.folder / "claim.json").read_bytes() == claim
    assert active.get(0) is tp and failures == {}
    assert c.publications == []
