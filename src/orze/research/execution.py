"""Small execution helpers for explicit, bounded research batches."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager


@contextmanager
def prepare_batch(items, prepare, *, workers):
    """Prepare a fixed batch concurrently and consume results as they finish.

    The caller evaluates results serially, and owns admission, costs and any
    subprocess cleanup. Every submitted callback is joined on exit, including
    early delivery or an exception; this helper never silently cancels work.
    Keep the batch inside the research controller's existing call budget.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    items = list(items)
    if len(items) > workers:
        raise ValueError("batch exceeds the explicit worker bound")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(prepare, item): item for item in items}
        yield ((futures[future], future.result()) for future in as_completed(futures))


def feature_tensor(values, *, device, dtype):
    """Convert stored arrays or existing tensors without a CUDA-to-NumPy hop.

    NumPy inputs are copied so a read-only memory map cannot back a writable
    tensor. Tensor inputs preserve autograd and avoid copying when compatible.
    PyTorch and NumPy remain optional, imported only when this helper is used.
    """
    import torch

    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    import numpy as np

    return torch.as_tensor(np.array(values, copy=True), device=device, dtype=dtype)


def run_prepared(spec, prepare, evaluate, *, output, policy=None,
                 finished=None, unused=None):
    """Run independent preparations concurrently, evaluating ready work serially.

    ``prepare(context)`` owns admitted model calls and returns an opaque value;
    ``evaluate(context, value)`` owns isolated execution and returns an outcome.
    Omitted ``plan.workers`` defaults to two; explicit limits are preserved and
    every preparation consumes one native action, including unused proposals.

    To stop on verified delivery, supply both ``finished()`` and
    ``unused(context, value)``. The evaluator must publish actual delivery before
    returning; unused returns a blocked, unscored outcome with the request's cost
    and provenance. All submitted preparations settle before return. Exceptions
    stop the rollout without retry; callbacks own child cleanup and reservations.
    """
    from orze.research.exploration import detached, run_online, validate_spec
    from orze.research.exploration_policies import ParallelRefine

    if not callable(prepare) or not callable(evaluate):
        raise ValueError("authorized prepare and evaluate callbacks are required")
    if ((finished is None) != (unused is None) or
            (finished is not None and (not callable(finished) or not callable(unused)))):
        raise ValueError("finished and unused callbacks must be supplied together")
    spec = detached(spec)
    if isinstance(spec, dict) and isinstance(spec.get("plan"), dict):
        spec["plan"].setdefault("workers", 2)
    spec = validate_spec(spec)
    selected = ParallelRefine() if policy is None else policy

    class UntilDelivered:
        def decide(self, view):
            if finished is not None and finished():
                return {"actions": [], "reason": "Verified research delivery completed"}
            return selected.decide(view)

    def execute(contexts):
        outcomes = {}
        with prepare_batch(contexts, prepare, workers=spec["plan"]["workers"]) as ready:
            for context, value in ready:
                if finished is not None and finished():
                    outcome = detached(unused(context, value))
                    if outcome.get("status") != "blocked" or outcome.get("score") is not None:
                        raise ValueError("unused proposals must be blocked and unscored")
                    outcome.setdefault("feedback", {})["unused_after_delivery"] = True
                else:
                    outcome = evaluate(context, value)
                outcomes[context["action"]["id"]] = outcome
        return outcomes

    return run_online(spec, UntilDelivered(), execute, output)
