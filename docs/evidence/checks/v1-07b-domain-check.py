#!/usr/bin/env python3
"""Independent stdlib-only scheduling verdict for fixed holdout evidence.

This does not import the application Domain, candidate generator or Core. It
checks declared feasibility/value only, never global optimality or process
closure. A supplied expected verdict may come from an archived evaluation;
passing here alone is not evidence that the product executed that evaluation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


if sys.flags.optimize:
    raise SystemExit("Run without -O: evidence assertions must be enabled.")

PROTOCOLS = ("schedule-feasibility-v1", "schedule-feasibility-v2")
MAX_CANDIDATE_BYTES = 16384


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _constant(_):
    raise ValueError("non-JSON numeric constant")


def strict_json(raw):
    if type(raw) is not bytes:
        raise ValueError("JSON must be bytes")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)


def validate_instance(instance):
    """Validate the input independently before considering candidate faults."""
    if type(instance) is not dict or set(instance) != {"instance_id", "horizon", "capacity", "jobs"}:
        raise ValueError("invalid instance fields")
    if (type(instance["instance_id"]) is not str or not instance["instance_id"]
            or type(instance["horizon"]) is not int or instance["horizon"] <= 0
            or type(instance["capacity"]) is not int or instance["capacity"] <= 0
            or type(instance["jobs"]) is not list):
        raise ValueError("invalid instance types")
    identifiers = []
    for job in instance["jobs"]:
        if type(job) is not dict or set(job) != {
                "id", "release", "deadline", "duration", "demand", "value", "required", "after"}:
            raise ValueError("invalid job fields")
        if (type(job["id"]) is not str or not job["id"]
                or type(job["required"]) is not bool or type(job["after"]) is not list
                or any(type(value) is not str for value in job["after"])
                or len(set(job["after"])) != len(job["after"])):
            raise ValueError("invalid job identity")
        for name in ("release", "deadline", "duration", "demand", "value"):
            if type(job[name]) is not int:
                raise ValueError("job integer field rejects bool and float")
        if (job["release"] < 0 or job["deadline"] < 0 or job["duration"] <= 0
                or job["demand"] <= 0 or job["value"] < 0):
            raise ValueError("invalid job quantity")
        identifiers.append(job["id"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate input job")
    if any(dependency not in identifiers for job in instance["jobs"] for dependency in job["after"]):
        raise ValueError("unknown input dependency")
    return instance


def evaluate(instance, raw, protocol):
    """Return exact typed verdict; only valid verdicts carry objective value."""
    validate_instance(instance)
    if type(protocol) is not str or protocol not in PROTOCOLS:
        raise ValueError("unsupported protocol")

    def invalid(reason):
        return {"status": "invalid", "reason_code": reason}

    if type(raw) is not bytes:
        return invalid("candidate_schema_invalid")
    if len(raw) > MAX_CANDIDATE_BYTES:
        return invalid("candidate_json_invalid")
    try:
        candidate = strict_json(raw)
    except (ValueError, UnicodeError, RecursionError):
        return invalid("candidate_json_invalid")
    if (type(candidate) is not dict or set(candidate) != {"instance_id", "schedule"}
            or type(candidate["instance_id"]) is not str
            or candidate["instance_id"] != instance["instance_id"]
            or type(candidate["schedule"]) is not list):
        return invalid("candidate_schema_invalid")
    entries = candidate["schedule"]
    # Whole-candidate schema precedes semantic checks. No JSON coercion of
    # bool/float start values, even if Python would compare True == 1.
    if any(type(item) is not dict or set(item) != {"job_id", "start"}
           or type(item["job_id"]) is not str or type(item["start"]) is not int
           or item["start"] < 0 for item in entries):
        return invalid("candidate_schema_invalid")
    jobs = {job["id"]: job for job in instance["jobs"]}
    selected = [entry["job_id"] for entry in entries]
    if any(identity not in jobs for identity in selected):
        return invalid("unknown_job")
    if any(count != 1 for count in Counter(selected).values()):
        return invalid("duplicate_job")
    if any(job["required"] and job["id"] not in selected for job in instance["jobs"]):
        return invalid("missing_required")
    spans = {entry["job_id"]: (entry["start"], entry["start"] + jobs[entry["job_id"]]["duration"])
             for entry in entries}
    if any(start < jobs[identity]["release"]
           or end > min(jobs[identity]["deadline"], instance["horizon"])
           for identity, (start, end) in spans.items()):
        return invalid("time_window")
    if any(dependency not in spans for identity in spans for dependency in jobs[identity]["after"]):
        return invalid("missing_prerequisite")
    if any(spans[dependency][1] > spans[identity][0]
           for identity in spans for dependency in jobs[identity]["after"]):
        return invalid("precedence_violation")
    capacity = instance["capacity"] if protocol == PROTOCOLS[0] else 1
    # Demand is constant between integer start/end events. At every such tick,
    # recompute active jobs directly with start <= tick < end; this is equivalent
    # to checking every integer tick, without sharing the producer's slot state.
    event_ticks = sorted({point for span in spans.values() for point in span})
    if any(sum(jobs[identity]["demand"] for identity, (start, end) in spans.items()
               if start <= tick < end) > capacity for tick in event_ticks):
        return invalid("capacity_overload")
    return {"status": "valid", "reason_code": "feasible",
            "scheduled_value": sum(jobs[identity]["value"] for identity in selected)}


def verify_result(instance, raw, protocol, result):
    expected = evaluate(instance, raw, protocol)
    # Canonical JSON equality keeps exact integer 0 distinct from False/0.0.
    encoded = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if encoded(result) != encoded(expected):
        raise ValueError("recorded result differs from independent feasibility verdict")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--expected-result", type=Path,
                        help="Optional archived verdict JSON; no process execution is inferred")
    args = parser.parse_args()
    instance_raw, candidate_raw = args.instance.read_bytes(), args.candidate.read_bytes()
    instance = strict_json(instance_raw)
    verdict = evaluate(instance, candidate_raw, args.protocol)
    if args.expected_result is not None:
        verify_result(instance, candidate_raw, args.protocol, strict_json(args.expected_result.read_bytes()))
    print(json.dumps({"candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "instance_file_sha256": hashlib.sha256(instance_raw).hexdigest(), "protocol": args.protocol,
        "verdict": verdict, "scope": "independent domain verdict only; no optimum or execution proof"},
        sort_keys=True))


if __name__ == "__main__":
    main()
