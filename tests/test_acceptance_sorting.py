"""Sorting oracle/trace and real private-FD transport tests, not CLI acceptance.

The end-to-end Policy/Orze/SQLite acceptance is owned by the shared harness.
These tests never insert an observation or use an external action runner.
"""
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import pytest

from examples.acceptance import common, sorting


def request(candidate, operation="measure", identities=()):
    return {"version": 1, "purpose": "check finite sorting behavior", "inputs": {},
            "timeout_seconds": 2, "outputs": copy.deepcopy(common.OUTPUTS),
            "input_artifact_ids": list(identities),
            "payload": {"candidate": candidate, "operation": operation}}


def prepared(dataset, candidate, operation="measure", bindings=()):
    domain = sorting.SortingDomain({"dataset": dataset})
    value = domain.prepare(request(candidate, operation, [b["artifact_id"] for b in bindings]),
                           tuple(bindings))
    return domain, value


def envelope(inputs):
    cost, details = sorting.measure(inputs["dataset"], inputs["candidate"])
    return common.write_result(inputs, cost, details, started_cpu=time.process_time(),
                               started_wall=time.monotonic())


@contextmanager
def sealed(raw):
    writer = os.memfd_create("sorting-unit-input", os.MFD_ALLOW_SEALING)
    reader = None
    try:
        assert os.write(writer, raw) == len(raw)
        fcntl.fcntl(writer, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL)
        reader = os.open("/proc/self/fd/" + str(writer), os.O_RDONLY | os.O_CLOEXEC)
        yield reader
    finally:
        if reader is not None:
            os.close(reader)
        os.close(writer)


@pytest.mark.parametrize("candidate,cost,status", [
    ("baseline", 9, "valid"), ("challenger", 0, "invalid"), ("unchecked", 29, "unknown"),
])
def test_default_measured_cost_and_scientific_status(tmp_path, monkeypatch, candidate, cost, status):
    monkeypatch.chdir(tmp_path)
    domain, run = prepared(sorting.DEFAULT_DATASET, candidate)
    result = envelope(run["action"]["inputs"])
    assert result["cost"] == cost == len(result["details"]["trace"])
    assert result["details"]["outputs"] == (
        sorting.DEFAULT_DATASET if candidate == "challenger"
        else [sorted(row) for row in sorting.DEFAULT_DATASET])
    claim, = domain.interpret(run, result)
    assert claim["validation"]["status"] == status
    assert claim["values"]["cost"] == cost
    assert claim["name"] == "objective"
    assert run["action"]["command"] == [sys.executable, str(Path(sorting.__file__).resolve())]


@pytest.mark.parametrize("candidate,cost,status", [
    ("baseline", 8, "valid"), ("challenger", 0, "valid"), ("unchecked", 29, "unknown"),
])
def test_counterfactual_changes_only_dataset(tmp_path, monkeypatch, candidate, cost, status):
    monkeypatch.chdir(tmp_path)
    original_domain, original = prepared(sorting.DEFAULT_DATASET, candidate)
    domain, run = prepared(sorting.COUNTERFACTUAL_DATASET, candidate)
    result = envelope(run["action"]["inputs"])
    assert result["cost"] == cost
    assert domain.interpret(run, result)[0]["validation"]["status"] == status
    assert original["action"]["command"] == run["action"]["command"]
    assert original["observation"]["protocol_fingerprint"] == run["observation"]["protocol_fingerprint"]
    assert sorting.DEFAULT_DATASET[:-1] == sorting.COUNTERFACTUAL_DATASET[:-1]
    assert sorting.DEFAULT_DATASET[-1] != sorting.COUNTERFACTUAL_DATASET[-1]
    assert result["details"]["outputs"] == [sorted(row) for row in sorting.COUNTERFACTUAL_DATASET]


@pytest.mark.parametrize("candidate,trace", [
    ("baseline", [[0, 3, 1, ">", True], [0, 3, 2, ">", True], [0, 1, 2, ">", False]]),
    ("unchecked", [[0, 1, 3, "<", True], [0, 2, 1, "<", False], [0, 2, 3, "<", True]]),
])
def test_handwritten_trace_includes_terminating_false(candidate, trace):
    cost, details = sorting.measure([[3, 1, 2]], candidate)
    assert details == {"outputs": [[1, 2, 3]], "trace": trace}
    assert cost == 3
    sorting.check_measurement([[3, 1, 2]], candidate, cost, details)


def test_oracle_checks_negatives_duplicates_and_empty_case():
    dataset = [[], [-1, -1, 2], [3, 0, -1, 3]]
    for candidate in ("baseline", "unchecked"):
        cost, details = sorting.measure(dataset, candidate)
        sorting.check_measurement(dataset, candidate, cost, details)
        assert details["outputs"] == [sorted(row) for row in dataset]
        assert sorting._oracle(dataset, details["outputs"])["valid"] is True
    assert sorting._oracle([[1, 1]], [[1]])["multiset_preserved"] is False
    assert sorting._oracle([[2, 1]], [[2, 1]])["nondecreasing"] is False


@pytest.mark.parametrize("fault", ["count", "operand", "boolean", "early_end", "output"])
def test_trace_checker_rejects_forged_measurement(fault):
    cost, details = sorting.measure(sorting.DEFAULT_DATASET, "baseline")
    if fault == "count":
        cost += 1
    elif fault == "operand":
        details["trace"][0][1] += 1
    elif fault == "boolean":
        details["trace"][0][4] = not details["trace"][0][4]
    elif fault == "early_end":
        details["trace"].pop()
        cost -= 1
    else:
        details["outputs"][-1].pop()
    with pytest.raises(ValueError):
        sorting.check_measurement(sorting.DEFAULT_DATASET, "baseline", cost, details)


def test_unchecked_measurement_does_not_sneak_in_a_scientific_oracle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    domain, run = prepared(sorting.DEFAULT_DATASET, "unchecked")
    result = envelope(run["action"]["inputs"])
    def forbidden(*args):
        pytest.fail("unchecked measure invoked scientific validity oracle")
    monkeypatch.setattr(sorting, "_oracle", forbidden)
    claim, = domain.interpret(run, result)
    assert claim["validation"]["status"] == "unknown"


@pytest.fixture
def source_envelopes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results, bindings, statuses = {}, [], {}
    for candidate in sorting.CANDIDATES:
        domain, run = prepared(sorting.DEFAULT_DATASET, candidate)
        result = envelope(run["action"]["inputs"])
        identity = "sorting-source-" + candidate
        results[identity] = result
        bindings.append({"artifact_id": identity, "content_sha256": common.digest(result)})
        statuses[candidate] = domain.interpret(run, result)
    return results, bindings, statuses


def test_analysis_reads_three_real_sealed_fds_without_rerunning_or_rewriting_unknown(
        source_envelopes, tmp_path, monkeypatch):
    from contextlib import ExitStack
    results, bindings, statuses = source_envelopes
    original_results, original_unknown = copy.deepcopy(results), copy.deepcopy(statuses["unchecked"])
    domain, run = prepared(sorting.DEFAULT_DATASET, "unchecked", "analyze", bindings)
    inputs = run["action"]["inputs"]
    def forbidden(*args):
        pytest.fail("analysis reran a sorting algorithm")
    monkeypatch.setattr(sorting, "measure", forbidden)
    monkeypatch.setitem(sys.modules, "common", common)
    with ExitStack() as stack:
        input_fd = stack.enter_context(sealed(common.canonical(inputs)))
        descriptors = {identity: stack.enter_context(sealed(common.canonical(result)))
                       for identity, result in results.items()}
        monkeypatch.setenv("ORZE_ACTION_INPUT_FD", str(input_fd))
        monkeypatch.setenv("ORZE_ACTION_SOURCE_FDS", json.dumps(descriptors))
        sorting.main()
    result = json.loads((tmp_path / "result.json").read_bytes())
    assert result["operation"] == "analyze"
    assert result["cost"] == 29 > results["sorting-source-baseline"]["cost"]
    assert result["source_artifact_ids"] == [item["artifact_id"] for item in bindings]
    assert [item["valid"] for item in result["details"]["source_checks"]] == [True, False, True]
    assert domain.interpret(run, result)[0]["validation"]["status"] == "valid"
    assert result["worker_cpu_seconds"] >= 0 and result["worker_wall_seconds"] >= 0
    assert results == original_results
    assert statuses["unchecked"] == original_unknown
    assert statuses["unchecked"][0]["validation"]["status"] == "unknown"


@pytest.mark.parametrize("fault", ["missing_source", "duplicate_candidate", "wrong_dataset", "bad_trace"])
def test_analysis_rejects_incomplete_or_inconsistent_source_measurements(source_envelopes, fault):
    results, bindings, _ = source_envelopes
    domain, run = prepared(sorting.DEFAULT_DATASET, "unchecked", "analyze", bindings)
    if fault == "missing_source":
        del results["sorting-source-baseline"]
    elif fault == "duplicate_candidate":
        results["sorting-source-baseline"]["candidate"] = "challenger"
    elif fault == "wrong_dataset":
        results["sorting-source-baseline"]["dataset_sha256"] = "0" * 64
    else:
        results["sorting-source-unchecked"]["details"]["trace"][0][4] = True
    with pytest.raises(ValueError):
        sorting.analyze(run["action"]["inputs"], results)


def test_actual_source_fd_bytes_must_match_captured_digest(source_envelopes, monkeypatch):
    from contextlib import ExitStack
    results, bindings, _ = source_envelopes
    domain, run = prepared(sorting.DEFAULT_DATASET, "unchecked", "analyze", bindings)
    with ExitStack() as stack:
        descriptors = {}
        for identity, result in results.items():
            changed = copy.deepcopy(result)
            if identity.endswith("unchecked"):
                changed["cost"] = 0
            descriptors[identity] = stack.enter_context(sealed(common.canonical(changed)))
        monkeypatch.setenv("ORZE_ACTION_SOURCE_FDS", json.dumps(descriptors))
        with pytest.raises(ValueError, match="actual source bytes"):
            common.read_source_results(run["action"]["inputs"])


def test_domain_config_and_dataset_are_detached_and_typed():
    config = {"dataset": copy.deepcopy(sorting.DEFAULT_DATASET)}
    domain = sorting.SortingDomain(config)
    config["dataset"][-1].clear()
    assert domain.dataset == sorting.DEFAULT_DATASET
    for invalid in ({}, {"dataset": [], "extra": 1}, {"dataset": [[True]]}, {"dataset": [1, 2]}):
        with pytest.raises(ValueError):
            sorting.SortingDomain(invalid)
