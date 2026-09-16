"""Finite-dataset sorting example; costs are actual comparator invocations.

This is application code, not a Core runner. The unchecked candidate publishes
an explicit unknown until a separate source-bound analysis validates its output.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time


DEFAULT_DATASET = [[], [5], [1, 1], [0, 1, 2, 2, 4, 3, 5, 6]]
COUNTERFACTUAL_DATASET = [[], [5], [1, 1], [0, 1, 2, 2, 3, 4, 5, 6]]
CANDIDATES = ("baseline", "challenger", "unchecked")
PROTOCOL = {
    "name": "sorting_comparisons", "version": 1,
    "metric": "comparator_calls_including_false",
}
ADAPTER_ID = "acceptance.sorting.v1"


def _dataset(value):
    if (type(value) is not list or not value
            or any(type(row) is not list or any(type(v) is not int for v in row) for row in value)):
        raise ValueError("sorting dataset must be a nonempty list of integer lists")
    return copy.deepcopy(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def measure(dataset, candidate):
    """Run the named algorithm, recording every value comparison, even False."""
    dataset = _dataset(dataset)
    if candidate not in CANDIDATES:
        raise ValueError("unknown sorting candidate")
    trace = []
    outputs = []
    for case, original in enumerate(dataset):
        values = list(original)

        def compare(left, right, relation):
            result = left > right if relation == ">" else left < right
            trace.append([case, left, right, relation, result])
            return result

        if candidate == "baseline":
            # Adjacent-exchange insertion sort, including each terminating False.
            for index in range(1, len(values)):
                cursor = index
                while cursor > 0:
                    if not compare(values[cursor - 1], values[cursor], ">"):
                        break
                    values[cursor - 1], values[cursor] = values[cursor], values[cursor - 1]
                    cursor -= 1
        elif candidate == "unchecked":
            for first in range(len(values) - 1):
                least = first
                for cursor in range(first + 1, len(values)):
                    if compare(values[cursor], values[least], "<"):
                        least = cursor
                values[first], values[least] = values[least], values[first]
        # Challenger deliberately returns its input without any comparisons.
        outputs.append(values)
    return len(trace), {"outputs": outputs, "trace": trace}


def check_measurement(dataset, candidate, cost, details):
    """Independently replay the recorded comparison decisions and full outputs.

    This checks instrumentation and candidate behavior, not scientific validity.
    Insertion replay uses a saved pivot and shifting, unlike the measured swaps.
    """
    dataset = _dataset(dataset)
    if (candidate not in CANDIDATES or type(cost) is not int or cost < 0
            or type(details) is not dict or set(details) != {"outputs", "trace"}
            or type(details["outputs"]) is not list or type(details["trace"]) is not list):
        raise ValueError("invalid sorting measurement structure")
    trace, outputs = details["trace"], details["outputs"]
    if cost != len(trace) or len(outputs) != len(dataset):
        raise ValueError("sorting comparator count or output membership changed")
    if any(type(row) is not list or any(type(v) is not int for v in row) for row in outputs):
        raise ValueError("sorting output must contain exact integers")
    position = 0

    def consume(case, left, right, relation):
        nonlocal position
        if position >= len(trace):
            raise ValueError("sorting comparison trace ended early")
        event = trace[position]
        position += 1
        result = left > right if relation == ">" else left < right
        if (type(event) is not list or len(event) != 5
                or any(type(event[i]) is not int for i in (0, 1, 2))
                or type(event[3]) is not str or type(event[4]) is not bool
                or event != [case, left, right, relation, result]):
            raise ValueError("sorting comparison trace does not match algorithm")
        return result

    for case, original in enumerate(dataset):
        values = list(original)
        if candidate == "baseline":
            for index in range(1, len(values)):
                pivot, cursor = values[index], index - 1
                while cursor >= 0:
                    if not consume(case, values[cursor], pivot, ">"):
                        break
                    values[cursor + 1] = values[cursor]
                    cursor -= 1
                values[cursor + 1] = pivot
        elif candidate == "unchecked":
            for first in range(len(values)):
                least = first
                for cursor in range(first + 1, len(values)):
                    if consume(case, values[cursor], values[least], "<"):
                        least = cursor
                replacement = values[least]
                values[least] = values[first]
                values[first] = replacement
        if values != outputs[case]:
            raise ValueError("sorting output differs from its recorded algorithm")
    if position != len(trace):
        raise ValueError("sorting comparison trace has trailing events")


def _oracle(dataset, outputs):
    """A separate finite-test oracle: preserve multiplicity and nondecreasing order."""
    same_elements = (len(dataset) == len(outputs)
                     and all(Counter(left) == Counter(right) for left, right in zip(dataset, outputs)))
    ordered = all(all(a <= b for a, b in zip(row, row[1:])) for row in outputs)
    return {"multiset_preserved": same_elements, "nondecreasing": ordered,
            "valid": same_elements and ordered}


def analyze(inputs, source_results):
    """Validate three actual source envelopes without rerunning an algorithm."""
    dataset = _dataset(inputs["dataset"])
    expected_ids = [binding["artifact_id"] for binding in inputs["source_bindings"]]
    if type(source_results) is not dict or set(source_results) != set(expected_ids) or len(expected_ids) != 3:
        raise ValueError("sorting analysis requires exactly three bound sources")
    selected = {}
    for artifact_id, result in source_results.items():
        fields = {"version", "candidate", "operation", "dataset_sha256", "cost", "details",
                  "source_artifact_ids", "worker_cpu_seconds", "worker_wall_seconds"}
        if (type(result) is not dict or set(result) != fields or result.get("version") != 1
                or type(result.get("version")) is not int
                or result.get("candidate") not in CANDIDATES
                or result.get("candidate") in selected
                or result.get("operation") != "measure"
                or result.get("dataset_sha256") != _digest(dataset)
                or result.get("source_artifact_ids") != []):
            raise ValueError("sorting analysis source identity mismatch")
        if any(type(result[key]) not in (int, float) or not math.isfinite(result[key])
               or result[key] < 0 for key in ("worker_cpu_seconds", "worker_wall_seconds")):
            raise ValueError("sorting source timing is invalid")
        candidate = result["candidate"]
        check_measurement(dataset, candidate, result.get("cost"), result.get("details"))
        selected[candidate] = (artifact_id, result)
    if set(selected) != set(CANDIDATES):
        raise ValueError("sorting analysis source candidates are incomplete")
    checks = []
    for candidate in CANDIDATES:
        artifact_id, result = selected[candidate]
        checks.append({"artifact_id": artifact_id, "candidate": candidate, "cost": result["cost"],
                       **_oracle(dataset, result["details"]["outputs"])})
    unchecked = selected["unchecked"][1]
    # This new evaluator occurrence republishes the checked metric/trace from
    # the source. It is not a new sorting measurement or statistical replica.
    return unchecked["cost"], {**copy.deepcopy(unchecked["details"]), "source_checks": checks}


class SortingDomain:
    def __init__(self, config):
        if type(config) is not dict or set(config) != {"dataset"}:
            raise ValueError("sorting domain config requires only dataset")
        self.dataset = _dataset(config["dataset"])

    def prepare(self, request, sources):
        from .common import prepare
        payload = request.get("payload") if type(request) is dict else None
        if (type(payload) is not dict or set(payload) != {"candidate", "operation"}
                or payload["candidate"] not in CANDIDATES
                or payload["operation"] not in ("measure", "analyze")
                or (payload["operation"] == "analyze" and payload["candidate"] != "unchecked")):
            raise ValueError("invalid sorting request payload")
        if ((payload["operation"] == "measure" and len(sources) != 0)
                or (payload["operation"] == "analyze" and len(sources) != 3)):
            raise ValueError("sorting source count does not match operation")
        return prepare(request, sources, dataset=self.dataset, worker_path=str(Path(__file__).resolve()),
                       adapter_id=ADAPTER_ID, protocol=PROTOCOL)

    def interpret(self, prepared, envelope):
        from .common import claim
        inputs = prepared["action"]["inputs"]
        details = envelope["details"]
        if inputs["operation"] == "measure":
            check_measurement(inputs["dataset"], inputs["candidate"], envelope["cost"], details)
            if inputs["candidate"] == "unchecked":
                return claim(prepared, envelope, status="unknown",
                             reason_code="sorting_validity_not_checked")
        else:
            if type(details) is not dict or set(details) != {"outputs", "trace", "source_checks"}:
                raise ValueError("sorting analysis result is incomplete")
            check_measurement(inputs["dataset"], "unchecked", envelope["cost"],
                              {key: details[key] for key in ("outputs", "trace")})
            checks = details["source_checks"]
            expected_ids = {item["artifact_id"] for item in inputs["source_bindings"]}
            if (type(checks) is not list or len(checks) != 3
                    or any(type(check) is not dict or set(check) != {
                        "artifact_id", "candidate", "cost", "multiset_preserved", "nondecreasing", "valid"}
                           for check in checks)
                    or {check.get("artifact_id") for check in checks} != expected_ids
                    or [check.get("candidate") for check in checks] != list(CANDIDATES)
                    or any(type(check[key]) is not bool for check in checks
                           for key in ("valid", "multiset_preserved", "nondecreasing"))
                    or any(check["valid"] != (check["multiset_preserved"] and check["nondecreasing"])
                           or type(check["cost"]) is not int or check["cost"] < 0 for check in checks)
                    or checks[-1].get("cost") != envelope["cost"]
                    or checks[-1].get("valid") is not True):
                raise ValueError("sorting analysis source checks are not complete")
        oracle = _oracle(inputs["dataset"], details["outputs"])
        return claim(prepared, envelope, status="valid" if oracle["valid"] else "invalid",
                     reason_code="sorting_finite_dataset_valid" if oracle["valid"]
                     else "sorting_finite_dataset_invalid")


def main():
    # Direct worker execution imports only this sibling stdlib module. Core is
    # never imported into the worker, and no extra runner is created here.
    from common import read_inputs, read_source_results, write_result
    started_cpu, started_wall = time.process_time(), time.monotonic()
    inputs = read_inputs()
    if inputs["operation"] == "measure":
        cost, details = measure(inputs["dataset"], inputs["candidate"])
    elif inputs["operation"] == "analyze" and inputs["candidate"] == "unchecked":
        cost, details = analyze(inputs, read_source_results(inputs))
    else:
        raise ValueError("unsupported sorting operation")
    write_result(inputs, cost, details, started_cpu=started_cpu, started_wall=started_wall)


if __name__ == "__main__":
    main()
