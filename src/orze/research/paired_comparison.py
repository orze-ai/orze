"""Compute explicit comparisons from a bound artifact of per-unit measurements.

Optional native CPU worker, not a selection policy or independence certificate.
Metric values are per-unit quantities whose arithmetic mean is meaningful under
the declared protocol. Ratios with different denominators need a domain adapter.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

MAX_BYTES = 2 * 1024 * 1024
MAX_UNITS = 10000


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise ValueError("finite JSON required") from exc


def _label(value):
    if type(value) is not str or not value or len(value) > 256:
        raise ValueError("nonempty bounded identity required")
    return value


def _config(value, depth=0):
    if depth > 16:
        raise ValueError("configuration nesting exceeds 16")
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("configuration keys must be strings")
        for child in value.values():
            _config(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _config(child, depth + 1)
    elif type(value) not in (type(None), str, bool, int, float):
        raise ValueError("JSON configuration required")


def _changes(reference, candidate, path=()):
    result = []
    for key in sorted(reference.keys() | candidate.keys()):
        left, right = key in reference, key in candidate
        a, b = reference.get(key), candidate.get(key)
        if left and right and type(a) is dict and type(b) is dict:
            result.extend(_changes(a, b, path + (key,)))
        elif not (left and right) or _json(a) != _json(b):
            result.append({"path": list(path + (key,)),
                           "reference": {"present": left, "value": a},
                           "candidate": {"present": right, "value": b}})
    return result


def _record(value):
    if type(value) is not dict or set(value) != {
            "id", "comparison_scope", "metric", "direction", "configuration", "units"}:
        raise ValueError("measurement record fields do not match the contract")
    for key in ("id", "comparison_scope", "metric"):
        _label(value[key])
    if value["direction"] not in ("minimize", "maximize"):
        raise ValueError("direction must be minimize or maximize")
    cfg = value["configuration"]
    if type(cfg) is not dict:
        raise ValueError("configuration must be an object")
    _config(cfg)
    if len(_json(cfg).encode()) > 8192:
        raise ValueError("configuration exceeds 8192 bytes")
    units = value["units"]
    if type(units) is not list or not 1 <= len(units) <= MAX_UNITS:
        raise ValueError("one to 10000 measurement units required")
    indexed = {}
    for unit in units:
        if type(unit) is not dict or set(unit) != {"id", "group", "value"}:
            raise ValueError("each unit requires id, group and value")
        key = _label(unit["id"])
        _label(unit["group"])
        number = unit["value"]
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError("finite numeric measurements required")
        if key in indexed:
            raise ValueError("duplicate measurement unit")
        indexed[key] = unit
    return indexed


def compare(document):
    """Return complete declared-config differences, paired means and known reuse.

    Unmatched scopes, units or group identities have no paired effect. Zero
    overlap in supplied prior uses does not prove that the history is complete.
    """
    if type(document) is not dict or set(document) != {"reference", "candidate", "prior_uses"}:
        raise ValueError("reference, candidate and prior_uses required")
    if len(_json(document).encode()) > MAX_BYTES:
        raise ValueError("comparison input exceeds two MiB")
    ref, cand = document["reference"], document["candidate"]
    a, b = _record(ref), _record(cand)
    changes = _changes(ref["configuration"], cand["configuration"])
    if len(changes) > 128:
        raise ValueError("more than 128 configuration differences")
    shared = a.keys() & b.keys()
    mismatched_groups = sum(a[key]["group"] != b[key]["group"] for key in shared)
    scope_fields = ("comparison_scope", "metric", "direction")
    same_scope = all(ref[key] == cand[key] for key in scope_fields)
    paired = same_scope and a.keys() == b.keys() and not mismatched_groups
    means = {"reference": math.fsum(row["value"] / len(a) for row in a.values()),
             "candidate": math.fsum(row["value"] / len(b) for row in b.values())}
    effect = None
    if paired:
        sign = 1 if ref["direction"] == "minimize" else -1
        deltas = {key: b[key]["value"] - a[key]["value"] for key in shared}
        groups = {}
        for key, delta in deltas.items():
            groups.setdefault(a[key]["group"], []).append(delta)
        gap = math.fsum(delta / len(deltas) for delta in deltas.values())
        effect = {"candidate_minus_reference_mean": gap,
                  "improvement": -sign * gap,
                  "relative_improvement": -sign * gap / abs(means["reference"]) if means["reference"] else None,
                  "candidate_better_units": sum(sign * x < 0 for x in deltas.values()),
                  "equal_units": sum(x == 0 for x in deltas.values()),
                  "candidate_worse_units": sum(sign * x > 0 for x in deltas.values()),
                  "groups": len(groups),
                  "equal_group_candidate_minus_reference_mean": math.fsum(
                      math.fsum(x / len(xs) for x in xs) / len(groups) for xs in groups.values())}
    prior = document["prior_uses"]
    overlap = None
    if prior is not None:
        if type(prior) is not list or len(prior) > 32:
            raise ValueError("prior_uses must be null or up to 32 declared records")
        overlap = []
        for record in prior:
            if type(record) is not dict or set(record) != {"id", "unit_ids"}:
                raise ValueError("prior use requires id and unit_ids")
            _label(record["id"])
            ids = record["unit_ids"]
            if type(ids) is not list or len(ids) > MAX_UNITS:
                raise ValueError("bounded prior-use unit list required")
            ids = {_label(key) for key in ids}
            overlap.append({"prior_use_id": record["id"], "prior_units": len(ids),
                            "reference_evaluation_overlap": len(ids & a.keys()),
                            "candidate_evaluation_overlap": len(ids & b.keys())})
    result = {"reference_id": ref["id"], "candidate_id": cand["id"],
              "comparison_scope": ref["comparison_scope"] if same_scope else None,
              "metric": ref["metric"] if same_scope else None,
              "direction": ref["direction"] if same_scope else None,
              "configuration_changes": changes,
              "configuration_meaning": "Declared JSON differences, including numeric representation; effective implementation equivalence is not inferred.",
              "pairing": {"complete": paired, "compatible_declared_scope": same_scope,
                          "reference_units": len(a), "candidate_units": len(b), "shared_units": len(shared),
                          "reference_only": len(a.keys() - b.keys()), "candidate_only": len(b.keys() - a.keys()),
                          "group_disagreements": mismatched_groups},
              "separate_unit_means": means, "paired_effect": effect, "prior_use_overlap": overlap,
              "interpretation": "Arithmetic on supplied measurements and identities only. Prior-use coverage is not certified complete. No population significance, causal attribution or independent-confirmation verdict is produced."}
    _json(result)  # Overflow is an invalid comparison, never an infinite score.
    return result


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def main():
    with os.fdopen(os.dup(int(os.environ["ORZE_ACTION_INPUT_FD"]))) as stream:
        inputs = json.load(stream)
    try:
        if type(inputs) is not dict or set(inputs) != {"source_id"}:
            raise ValueError("one bound source_id required")
        sources = json.loads(os.environ["ORZE_ACTION_SOURCE_FDS"])
        with os.fdopen(os.dup(sources[inputs["source_id"]]), "rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("comparison input exceeds two MiB")
        result = compare(json.loads(raw, object_pairs_hook=_unique))
        valid = result["pairing"]["complete"]
        status, reason = ("valid", "complete_paired_arithmetic") if valid else ("invalid", "incomparable_measurements")
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        result = {"error": str(exc)[:240]}
        status, reason = "invalid", "comparison_input_invalid"
    observation = {"name": "paired_comparison", "values": result,
                   "validation": {"status": status, "reason_code": reason},
                   "comparison_scope": result.get("comparison_scope")}
    Path("evaluation.json").write_text(_json({"version": 1, "observations": [observation]}))


if __name__ == "__main__":
    main()
