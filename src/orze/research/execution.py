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
