"""Existing public reservation safety, using only task-local files and threads.

No evaluator is run. The age fault changes only our held lock's time metadata
and mtime; it is not a 301-second wait or a claim of naturally elapsed time.
The silent-write fault is limited to the actual exposure ledger's device/inode.
"""
import hashlib
import json
import os
import threading

import pytest

from orze.core import benchmark_contract as benchmark


@pytest.fixture
def reservation(tmp_path):
    script = tmp_path / "eval_exact.py"
    script.write_text("# Never executed by these reservation tests.\n", encoding="utf-8")
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    cfg = {
        "_project_root": str(tmp_path),
        "eval_script": script.name,
        "eval_output": "eval_report.json",
        "sealed_hashes": {script.name: digest},
        "report": {
            "primary_metric": "avg_score", "sort": "ascending",
            "min_datasets": 2,
            "columns": [{"key": key, "label": key}
                        for key in ("avg_score", "metric_a", "metric_b")],
            "benchmark_contract": {
                "benchmark_id": "owner/benchmark", "revision": "a" * 40,
                "view": "default", "required_metrics": ["metric_a", "metric_b"],
                "receipt": "benchmark_receipt.json",
                "model_form": "single_model_single_pass",
                "evidence_scope": "local_reproduction",
                "selection_mode": "confirmation", "prior_exposures": 0,
                "max_evaluations": 1, "aggregate": "macro_mean",
                "aggregate_tolerance": 1e-9, "evaluator_sha256": digest,
                "dataset_manifest_sha256": "d" * 64, "scorer_sha256": "e" * 64,
            },
        },
    }
    assert benchmark.validate_benchmark_contract_config(cfg) == []
    ideas = [tmp_path / "results" / f"idea-{index}" for index in range(2)]
    for idea in ideas:
        idea.mkdir(parents=True)
    return cfg, ideas, benchmark.benchmark_exposure_ledger_path(cfg)


def _reserve(idea, cfg):
    try:
        return {"env": benchmark.prepare_benchmark_evaluation(idea, cfg), "error": None}
    except (benchmark.BenchmarkContractError, OSError) as error:
        # Both explicit public refusals and filesystem uncertainty deny launch.
        return {"env": None, "error": type(error).__name__ + ":" + str(error)}


@pytest.mark.parametrize("aged", [False, True], ids=["fresh-control", "live-owner-aged"])
def test_live_reservation_owner_cannot_be_displaced(reservation, monkeypatch, aged):
    cfg, ideas, ledger = reservation
    waiting, release = threading.Event(), threading.Event()
    real_open = os.open
    results, faults = {}, []

    def paused_open(path, flags, *args, **kwargs):
        if (threading.current_thread() is first
                and os.fspath(path) == str(ledger) and flags & os.O_APPEND):
            waiting.set()
            if not release.wait(10):
                raise AssertionError("fixture failed to release its own reservation thread")
        return real_open(path, flags, *args, **kwargs)

    def run(index):
        try:
            results[index] = _reserve(ideas[index], cfg)
        except BaseException as error:
            faults.append(error)

    first = threading.Thread(target=run, args=(0,), name="owned-reservation-first")
    second = threading.Thread(target=run, args=(1,), name="owned-reservation-second")
    monkeypatch.setattr(benchmark.os, "open", paused_open)
    second_started = False
    first.start()
    try:
        assert waiting.wait(10), (results, faults)
        assert first.is_alive()
        lock_dir = ledger.parent / benchmark.EXPOSURE_LOCK_DIR
        metadata_path = lock_dir / "lock.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        owner = metadata["host"], metadata["pid"]
        assert owner[1] == os.getpid()
        if aged:
            # The legacy lock consults metadata time, not directory mtime.
            # Preserve the real live owner's host/PID; mutate no other scope.
            metadata["time"] -= 301
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            os.utime(lock_dir, (metadata["time"], metadata["time"]))
            observed = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert (observed["host"], observed["pid"]) == owner
        second.start()
        second_started = True
        second.join(10)
        assert not second.is_alive(), "second reservation did not return within fixture bound"
    finally:
        release.set()
        first.join(10)
        if second_started:
            second.join(10)
    assert not first.is_alive() and not second.is_alive()
    assert not faults, faults
    assert set(results) == {0, 1}
    granted = [item["env"] for item in results.values() if item["env"] is not None]
    records = ([json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
               if ledger.exists() else [])
    assert len(granted) <= 1, {
        "returned_environments": len(granted), "ordinals": [r["exposure_ordinal"] for r in records],
        "outcomes": results,
    }
    if not aged:
        assert len(granted) == 1
        assert len(records) == 1
        assert results[1]["env"] is None


@pytest.mark.parametrize("silent", [False, True], ids=["durable-control", "silent-ledger-write"])
def test_nonce_requires_a_durable_exposure_record(reservation, monkeypatch, silent):
    cfg, ideas, ledger = reservation
    real_write = os.write
    intercepted = []

    def ledger_write(fd, data):
        descriptor = os.fstat(fd)
        try:
            named = ledger.stat()
        except FileNotFoundError:
            return real_write(fd, data)
        if (descriptor.st_dev, descriptor.st_ino) == (named.st_dev, named.st_ino):
            intercepted.append(len(data))
            if silent:
                return len(data)
        return real_write(fd, data)

    monkeypatch.setattr(benchmark.os, "write", ledger_write)
    result = _reserve(ideas[0], cfg)
    assert intercepted, "fault/control must reach the actual ledger descriptor"
    if silent:
        assert ledger.read_bytes() == b""
        assert result["env"] is None, {
            "returned_environment": result["env"], "durable_ledger_bytes": 0,
        }
        assert not (ideas[0] / benchmark.PROVENANCE_FILE).exists()
    else:
        assert result["error"] is None
        assert result["env"]["ORZE_BENCHMARK_EXPOSURE_ORDINAL"] == "1"
        record, = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        assert result["env"]["ORZE_BENCHMARK_EXPOSURE_RECORD_SHA256"] == record["record_sha256"]
        assert _reserve(ideas[1], cfg)["env"] is None
