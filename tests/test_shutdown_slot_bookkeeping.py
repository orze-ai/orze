"""Legacy shutdown mapping boundaries, not process-closure evidence.

Real GpuSlotManager bookkeeping and dicts; explicit native-stop result doubles,
fake tracked objects, managed=True. No worker, signal, GPU query, or Lake I/O.
"""
from types import SimpleNamespace

import pytest

from orze.engine import lifecycle, shutdown_publication
from orze.engine.gpu_slots import GpuSlotManager


def _tracked(name, *, held=False, on_close=None):
    closed = []

    def close_log():
        closed.append(name)
        if on_close is not None:
            on_close()

    return SimpleNamespace(
        idea_id=name, process=SimpleNamespace(pid=0, poll=lambda: 0),
        _termination_unconfirmed=held, close_log=close_log, closed=closed)


def _mapping(kind):
    # Public string-key assignment records composite GPU slots without asking
    # hardware/capacity permission. We are testing removal, not admission.
    return {} if kind == "dict" else GpuSlotManager([2, 3], mode="vram")


def _case(kind):
    training, evaluation = _mapping(kind), _mapping(kind)
    originals = {}
    for phase, mapping in (("training", training), ("evaluation", evaluation)):
        for key, held in (("2:10+3:11", False), ("2:12+3:13", True)):
            tracked = _tracked(f"{phase}-{'held' if held else 'closed'}", held=held)
            mapping[key] = tracked
            originals[(phase, key)] = tracked
    return training, evaluation, originals


def _assert_slots(mapping, expected):
    assert list(mapping.keys()) == list(expected)
    assert all(mapping.get(key) is value for key, value in expected.items())
    if isinstance(mapping, GpuSlotManager):
        expected_jobs = {gpu: [part for key in expected for part in key.split("+")
                               if int(part.split(":")[0]) == gpu]
                         for gpu in (2, 3)}
        assert mapping._gpu_jobs == expected_jobs
        assert mapping.total_used_slots == len(expected)
        assert mapping.gpu_ids_in_use() == {gpu for gpu, jobs in expected_jobs.items() if jobs}
        assert [mapping.job_count(gpu) for gpu in (2, 3)] == [len(expected_jobs[gpu]) for gpu in (2, 3)]
        assert mapping._next_id == 0, "shutdown must not reassign retained slots"


def _install_stops(monkeypatch, calls, callback=None):
    def stop(tracked, results_dir, phase, stopper, **kwargs):
        calls.append((phase, tracked))
        if callback is not None:
            callback(tracked)
        return not tracked._termination_unconfirmed

    def forbidden(*args, **kwargs):
        pytest.fail("mapping test must not enter legacy OS stop/reaper fallback")

    monkeypatch.setattr(shutdown_publication, "handle_shutdown", stop)
    monkeypatch.setattr(lifecycle, "_stop_for_shutdown", forbidden)
    monkeypatch.setattr(lifecycle, "terminate_role_process", forbidden)


def _shutdown(tmp_path, training, evaluation, *, kill, lake=None):
    lifecycle.graceful_shutdown(
        tmp_path, {}, training, evaluation, {}, 1, {}, lake,
        "fixture-host", "fixture-instance", kill_all=kill, managed=True)


def _assert_processed(originals, calls, *, kill):
    expected = [(phase, tracked) for (phase, _), tracked in originals.items()
                if kill or phase == "evaluation"]
    assert calls == expected, "only objects captured on shutdown entry may be stopped"
    for (phase, _), tracked in originals.items():
        expected_closes = ([tracked.idea_id] if phase == "training" and not kill
                           and not tracked._termination_unconfirmed else [])
        assert tracked.closed == expected_closes


@pytest.mark.parametrize("kind", ["dict", "slots"])
@pytest.mark.parametrize("kill", [False, True], ids=["detach", "kill"])
def test_closed_entries_removed_and_hold_slots_preserved(tmp_path, monkeypatch, kind, kill):
    training, evaluation, originals = _case(kind)
    calls = []
    _install_stops(monkeypatch, calls)
    _shutdown(tmp_path, training, evaluation, kill=kill)
    _assert_processed(originals, calls, kill=kill)
    for phase, mapping in (("training", training), ("evaluation", evaluation)):
        _assert_slots(mapping, {"2:12+3:13": originals[(phase, "2:12+3:13")]})


def _replace_and_add(training, evaluation):
    expected = {}
    for phase, mapping in (("training", training), ("evaluation", evaluation)):
        expected[phase] = {}
        for key in ("2:10+3:11", "2:12+3:13", "2:20+3:21"):
            replacement = _tracked(f"replacement-{phase}-{key}")
            mapping[key] = replacement
            expected[phase][key] = replacement
    return expected


@pytest.mark.parametrize("kind", ["dict", "slots"])
@pytest.mark.parametrize("kill", [False, True], ids=["detach", "kill"])
def test_callback_changes_do_not_expand_entry_snapshot(tmp_path, monkeypatch, kind, kill):
    training, evaluation, originals = _case(kind)
    calls, expected = [], {}
    first = originals[("training", "2:10+3:11")]

    def mutate(tracked=None):
        if not expected:
            expected.update(_replace_and_add(training, evaluation))

    if not kill:
        # Detach uses close_log, not handle_shutdown; mutate at that real seam.
        original_close = first.close_log

        def close_and_mutate():
            original_close()
            mutate()

        first.close_log = close_and_mutate
    _install_stops(monkeypatch, calls, mutate if kill else None)
    _shutdown(tmp_path, training, evaluation, kill=kill)
    _assert_processed(originals, calls, kill=kill)
    assert expected, "fixture mutation must occur inside shutdown"
    for phase, mapping in (("training", training), ("evaluation", evaluation)):
        _assert_slots(mapping, expected[phase])
        assert all(not tracked.closed for tracked in expected[phase].values())


@pytest.mark.parametrize("kind", ["dict", "slots"])
def test_final_cleanup_preserves_replacements_including_old_hold_key(tmp_path, monkeypatch, kind):
    training, evaluation, originals = _case(kind)
    calls, expected, lake_closed = [], {}, []

    def close_lake():
        lake_closed.append(True)
        expected.update(_replace_and_add(training, evaluation))

    _install_stops(monkeypatch, calls)
    _shutdown(tmp_path, training, evaluation, kill=True,
              lake=SimpleNamespace(close=close_lake))
    assert lake_closed == [True]
    _assert_processed(originals, calls, kill=True)
    for phase, mapping in (("training", training), ("evaluation", evaluation)):
        _assert_slots(mapping, expected[phase])
        assert all(not tracked.closed for tracked in expected[phase].values())
