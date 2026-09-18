"""Reviewed analysis and prediction for an application with active experiments.

The application supplies the experiment runner and hides its evaluation data.
Source checks are a review aid, not a sandbox. No simulator is exposed here.
"""
import copy
import json
import math
import time

import numpy as np

from .open_experiment import check_source


def protocols(value):
    """Validate the declared controllable inputs before running an experiment."""
    if type(value) is not list or not 1 <= len(value) <= 3:
        raise ValueError("provide one to three protocols")
    for protocol in value:
        if type(protocol) is not dict or set(protocol) != {"u"}:
            raise ValueError("each protocol requires exactly u")
        u = protocol["u"]
        if type(u) is not list or not 1 <= len(u) <= 80:
            raise ValueError("u requires one to eighty one-second inputs")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in u):
            raise ValueError("inputs must be finite numbers from zero through one")
    return copy.deepcopy(value)


def analyze(source, observations, history, seed=1729):
    code = compile(check_source(source, "analyze"), "reviewed-lab-analysis", "exec")
    namespace = {}
    start = time.process_time()
    exec(code, namespace)
    result = namespace["analyze"](copy.deepcopy(observations), copy.deepcopy(history), seed)
    raw = json.dumps(result, allow_nan=False, sort_keys=True)
    if len(raw.encode()) > 8192:
        raise ValueError("analysis findings exceed 8192 UTF-8 bytes")
    return {"findings": json.loads(raw), "process_cpu_seconds": time.process_time() - start}


def predict(source, observations, plans, seed=1729):
    """Pass experiment inputs, never reference outputs, to the frozen predictor."""
    code = compile(check_source(source, "predict"), "reviewed-lab-predictor", "exec")
    namespace = {}
    exec(code, namespace)
    raw = namespace["predict"](copy.deepcopy(observations), copy.deepcopy(plans), seed)
    if not isinstance(raw, (list, tuple)) or len(raw) != len(plans):
        raise ValueError("return one output vector per supplied protocol")
    result = []
    for p, values in zip(plans, raw):
        y = np.asarray(values, dtype=float)
        if y.shape != (len(p["u"]),) or not np.all(np.isfinite(y)):
            raise ValueError("return one finite prediction per input time")
        result.append(y.tolist())
    return result


def score(prediction, reference):
    """Equal-weight protocol RMSE; retain each protocol's error."""
    if len(prediction) != len(reference) or not reference:
        raise ValueError("prediction and reference protocols must match")
    errors = []
    for p, y in zip(prediction, reference):
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
        if p.shape != y.shape or p.ndim != 1 or not len(p) or not np.all(np.isfinite(p)):
            raise ValueError("prediction and reference outputs must match")
        errors.append(float(np.sqrt(np.mean((p-y)**2))))
    return {"loss": float(np.mean(errors)), "protocol_rmse": errors}
