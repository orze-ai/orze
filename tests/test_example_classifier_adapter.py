"""Smoke test for the toy ``example_classifier`` post-hoc adapter."""

from pathlib import Path

import numpy as np
import pytest


def test_example_classifier_registers():
    """Importing the adapter should self-register via @register_adapter."""
    from orze.engine.posthoc_adapters import example_classifier  # noqa: F401
    from orze.engine.posthoc_runner import list_adapters
    assert "example_classifier" in list_adapters()


def test_example_classifier_round_trip(tmp_path):
    """Round-trip: write NPZ → adapter loads, scores, returns metrics."""
    from orze.engine.posthoc_adapters import example_classifier
    from orze.engine.posthoc_runner import get_adapter

    n = 20
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(n, 3)).astype(np.float32)
    labels = rng.integers(0, 3, size=n)
    split = np.array(["public"] * 10 + ["private"] * 10)

    npz_path = tmp_path / "predictions.npz"
    np.savez(npz_path, logits=logits, labels=labels, split=split)

    cfg = {"posthoc": {"example_classifier": {"npz_path": str(npz_path)}}}
    run = get_adapter("example_classifier")
    metrics = run("idea-test", cfg, tmp_path)

    assert metrics["_adapter"] == "example_classifier"
    for key in ("acc_public", "acc_private", "acc_all"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_example_classifier_missing_file(tmp_path):
    """Missing NPZ → adapter should raise."""
    from orze.engine.posthoc_adapters import example_classifier
    from orze.engine.posthoc_runner import get_adapter

    cfg = {"posthoc": {"example_classifier": {"npz_path": str(tmp_path / "nope.npz")}}}
    run = get_adapter("example_classifier")
    with pytest.raises(FileNotFoundError):
        run("idea-test", cfg, tmp_path)
