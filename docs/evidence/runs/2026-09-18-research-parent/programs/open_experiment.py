"""Reviewed analysis and method code for an application-owned research problem.

The caller owns data access, code review, execution limits and provenance.
These syntax checks are not a sandbox. Diagnostics are not scored methods.
"""
import ast
import copy
import hashlib
import json
import time
import warnings

import numpy as np


def identity(action):
    return hashlib.sha256(json.dumps(action, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def check_source(source, entry):
    if type(source) is not str or not 1 <= len(source.encode()) <= 32768:
        raise ValueError("source must contain 1..32768 UTF-8 bytes")
    tree = ast.parse(source)
    prohibited = {"open", "eval", "exec", "compile", "__import__", "globals",
                  "locals", "getattr", "setattr", "vars", "input", "breakpoint"}
    io_names = {"load", "save", "loadtxt", "savetxt", "fromfile", "tofile",
                "memmap", "dump", "load_library", "ctypes", "os",
                "sys", "subprocess"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            if any(n.split(".")[0] not in {"numpy", "sklearn", "scipy", "math", "json", "collections"}
                   for n in names) or getattr(node, "level", 0):
                raise ValueError("only numpy, sklearn, scipy, math, json and collections imports supported")
        if isinstance(node, ast.Name) and node.id in prohibited:
            raise ValueError("dynamic execution, reflection and I/O unsupported")
        if isinstance(node, ast.Attribute) and (
                node.attr.startswith("_") or node.attr in io_names
                or node.attr.startswith(("fetch_", "read_", "write_"))):
            raise ValueError("private/reflection and I/O attributes unsupported")
    if not any(isinstance(n, ast.FunctionDef) and n.name == entry for n in tree.body):
        raise ValueError(f"define {entry}")
    return tree


def score(prediction, target):
    """Recompute equal-batch macro error, retaining absent-class coverage."""
    y = np.asarray(target["y"], dtype=int)
    raw = np.asarray(prediction)
    if raw.shape != y.shape or not np.all(np.isfinite(raw)):
        raise ValueError("return one finite class ID per evaluation row")
    if not np.all(np.isin(raw, np.arange(1, 7))):
        raise ValueError("class IDs must be integers from 1 through 6")
    pred = raw.astype(int)
    batch = np.asarray(target["batch"])
    batches = []
    for b in np.unique(batch):
        use = batch == b
        conf = np.zeros((6, 6), dtype=int)
        np.add.at(conf, (y[use] - 1, pred[use] - 1), 1)
        count = conf.sum(axis=1)
        recalls = [float(conf[c, c] / count[c]) if count[c] else None for c in range(6)]
        balanced_error = float(1 - np.mean([v for v in recalls if v is not None]))
        batches.append({"batch": int(b), "rows": int(use.sum()),
                        "balanced_error": balanced_error,
                        "error": float(np.mean(pred[use] != y[use])),
                        "class_counts": count.tolist(), "class_recall": recalls,
                        "confusion": conf.tolist()})
    return {"loss": float(np.mean([b["balanced_error"] for b in batches])),
            "error": float(np.mean(pred != y)), "batches": batches}


def execute(action, data, history, split="development", seed=1729):
    """Return measured facts and row-bound predictions for one reviewed action."""
    if type(action) is not dict or set(action) != {"kind", "source"}:
        raise ValueError("action requires exactly kind and source")
    kind = action["kind"]
    if kind not in {"analyze", "method"}:
        raise ValueError("choose analyze or method")
    if split not in {"development", "confirmation"}:
        raise ValueError("invalid evaluation split")
    if kind == "analyze" and split != "development":
        raise ValueError("analysis cannot read confirmation data")
    entry = "analyze" if kind == "analyze" else "fit_predict"
    code = compile(check_source(action["source"], entry), "reviewed-open-research", "exec")
    start, cpu = time.monotonic(), time.process_time()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if kind == "analyze":
            namespace = {}
            exec(code, namespace)
            # The analysis view cannot contain later-batch features or labels.
            if set(data) != {"train", "development"}:
                raise ValueError("analysis requires the early-data view only")
            finding = namespace[entry](copy.deepcopy(data), copy.deepcopy(history), seed)
            encoded = json.dumps(finding, sort_keys=True, allow_nan=False)
            if len(encoded.encode()) > 16384:
                raise ValueError("analysis result exceeds 16384 UTF-8 bytes")
            facts, predictions = {"findings": json.loads(encoded)}, None
        else:
            target = data[split]
            X, batch = np.asarray(target["X"], dtype=float), np.asarray(target["batch"])
            pred = np.empty(len(X), dtype=float)
            for b in np.unique(batch):
                use = batch == b
                namespace = {}
                exec(code, namespace)
                # No target labels/concentration/row IDs enter method code.
                local = np.asarray(namespace[entry](copy.deepcopy(data["train"]),
                                   X[use].copy(), batch[use].copy(), seed))
                if local.shape != (int(use.sum()),):
                    raise ValueError("prediction shape does not match evaluation batch")
                pred[use] = local
            facts = score(pred, target)
            predictions = {"row_ids": target["row_ids"], "prediction": pred.astype(int).tolist()}
    facts.update(kind=kind, action_id=identity(action), split=split,
                 worker_seconds=time.monotonic() - start,
                 process_cpu_seconds=time.process_time() - cpu,
                 warnings=[str(w.message)[:300] for w in seen[:8]])
    return facts, predictions
