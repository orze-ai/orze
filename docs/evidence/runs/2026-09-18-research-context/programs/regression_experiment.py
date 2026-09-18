"""Reviewed regression research with optional findings from the fitted method.

The application controls data splits, source review, and runtime limits. Source
checks are a review aid, not a sandbox. Findings are authored analysis outputs;
prediction quality is measured independently below.
"""
import copy
import json
import time

import numpy as np

from .open_experiment import check_source


def findings(value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False)
    if len(raw.encode()) > 8192:
        raise ValueError("findings exceed 8192 UTF-8 bytes")
    return json.loads(raw)


def score(prediction, target):
    pred = np.asarray(prediction, dtype=float)
    y = np.asarray(target["y"], dtype=float)
    if pred.ndim != 1 or pred.shape != y.shape or not pred.size:
        raise ValueError("return one prediction per evaluation row")
    if not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError("predictions and targets must be finite")
    groups = np.asarray(target["groups"])
    if groups.shape != y.shape:
        raise ValueError("one group per target is required")
    group_mae = [float(np.mean(np.abs(pred[groups == g] - y[groups == g])))
                 for g in np.unique(groups)]
    return {"loss": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mae": float(np.mean(np.abs(pred - y))),
            "group_mae": float(np.mean(group_mae)),
            "rows": int(y.size), "groups": len(group_mae)}


def execute(action, data, history, split="development", report_findings=False,
            input_fields=("X", "C", "feature_names", "elements")):
    if split not in {"development", "confirmation"}:
        raise ValueError("invalid evaluation split")
    if set(input_fields) & {"y", "groups", "row_ids"}:
        raise ValueError("evaluation labels and identifiers cannot be method inputs")
    if action.get("kind") == "analyze" and set(data) != {"train", "development"}:
        raise ValueError("analysis requires the early-data view only")
    kind = action["kind"]
    entry = "analyze" if kind == "analyze" else "fit_predict"
    if kind not in ("analyze", "method") or set(action) != {"kind", "source"}:
        raise ValueError("provide an analysis or method source")
    ns = {}
    code = compile(check_source(action["source"], entry), "reviewed-regression", "exec")
    start = time.process_time()
    exec(code, ns)
    if kind == "analyze":
        if split != "development":
            raise ValueError("analysis cannot access confirmation")
        result = ns[entry](copy.deepcopy(data), copy.deepcopy(history), 1729)
        return {"kind": kind, "findings": findings(result),
                "process_cpu_seconds": time.process_time() - start}, None
    train = copy.deepcopy(data["train"])
    target = data[split]
    inputs = {key: copy.deepcopy(target[key]) for key in input_fields}
    result = ns[entry](train, inputs, 1729)
    notes = None
    if report_findings and type(result) is dict:
        if set(result) != {"prediction", "findings"}:
            raise ValueError("method result requires prediction and findings")
        notes = findings(result["findings"])
        result = result["prediction"]
    pred = np.asarray(result, dtype=float)
    measured = score(pred, target)
    facts = {"kind": kind, **measured,
             "process_cpu_seconds": time.process_time() - start}
    if notes is not None:
        facts["findings"] = notes
    return facts, {"row_ids": target["row_ids"], "prediction": pred.tolist()}
