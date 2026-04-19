"""Tests for orze.engine.aggregations registry (F11)."""

import numpy as np
import pytest

from orze.engine.aggregations import (
    REGISTRY,
    list_aggregations,
    list_calibrators,
    make_aggregation,
    make_calibrator,
)


def test_registry_has_expected_aggregations():
    names = set(list_aggregations())
    assert {"last", "mean", "max", "noisy_or", "top_k_mean",
            "exp_weighted", "dense_late_softmax"} <= names
    for k in range(1, 6):
        assert f"late_k{k}" in names


def test_registry_has_expected_calibrators():
    names = set(list_calibrators())
    assert {"identity", "platt", "isotonic",
            "group_calibrated", "cv_mix"} <= names


@pytest.mark.parametrize("name", ["last", "mean", "max", "late_k1", "late_k3",
                                   "noisy_or"])
def test_deterministic_aggregations_produce_finite_scalar(name):
    agg = make_aggregation(name)
    probs = np.array([0.1, 0.2, 0.8, 0.3, 0.9])
    out = agg.apply(probs)
    assert np.isfinite(out)
    assert 0.0 <= out <= 1.0


def test_last_and_max_semantics():
    p = np.array([0.1, 0.9, 0.3])
    assert abs(make_aggregation("last").apply(p) - 0.3) < 1e-9
    assert abs(make_aggregation("max").apply(p) - 0.9) < 1e-9


def test_top_k_mean_respects_k():
    agg = make_aggregation("top_k_mean", k=2)
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # top-2 mean = (0.4+0.5)/2 = 0.45
    assert abs(agg.apply(p) - 0.45) < 1e-9


def test_exp_weighted_favors_recent():
    a = make_aggregation("exp_weighted", alpha=0.5)
    p_end_high = np.array([0.0, 0.0, 0.0, 1.0])
    p_start_high = np.array([1.0, 0.0, 0.0, 0.0])
    assert a.apply(p_end_high) > a.apply(p_start_high)


def test_noisy_or_monotone():
    a = make_aggregation("noisy_or")
    assert a.apply(np.array([0.5])) == pytest.approx(0.5, rel=1e-6)
    assert a.apply(np.array([0.5, 0.5])) > 0.5
    assert a.apply(np.array([0.0])) == pytest.approx(0.0, abs=1e-5)


def test_platt_calibrator_fits_and_monotone():
    cal = make_calibrator("platt")
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 200)
    y = (x + rng.normal(0, 0.3, 200) > 0).astype(float)
    cal.fit(x, y, n_iter=500, lr=0.2)
    assert cal.is_fit
    xs = np.linspace(-3, 3, 20)
    ys = cal.apply(xs)
    assert np.all(np.diff(ys) >= -1e-9)  # monotone non-decreasing
    assert 0.0 <= ys.min() and ys.max() <= 1.0


def test_isotonic_calibrator_monotone():
    cal = make_calibrator("isotonic")
    x = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.9, 0.7])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    cal.fit(x, y)
    xs = np.linspace(0, 1, 20)
    ys = cal.apply(xs)
    assert np.all(np.diff(ys) >= -1e-9)


def test_group_calibrated_rank_normalizes_per_group():
    cal = make_calibrator("group_calibrated")
    scores = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = cal.apply(scores, groups=groups)
    # within each group, the ranks should map to [0, 0.5, 1]
    assert np.allclose(out[:3], [0.0, 0.5, 1.0])
    assert np.allclose(out[3:], [0.0, 0.5, 1.0])


def test_identity_calibrator_is_noop():
    cal = make_calibrator("identity")
    x = np.array([0.1, 0.5, 0.9])
    assert np.allclose(cal.apply(x), x)


def test_cv_mix_fits_two_parts_and_applies():
    cal = make_calibrator("cv_mix")
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    good = y + rng.normal(0, 0.2, n)   # strongly informative
    noisy = rng.normal(0, 1, n)        # pure noise
    cal.fit([good, noisy], y, k_folds=5)
    assert cal.is_fit
    # alpha should land near 1.0 (strongly favour good part)
    assert cal.alpha >= 0.7
    out = cal.apply([good, noisy])
    assert out.shape == (n,)


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        make_aggregation("totally_not_a_thing")
    with pytest.raises(KeyError):
        make_calibrator("totally_not_a_thing")


def test_dense_late_softmax_runs():
    a = make_aggregation("dense_late_softmax", t=0.5)
    p = np.array([0.1, 0.2, 0.8])
    v = a.apply(p)
    assert 0.0 <= v <= 1.0


def test_every_registry_entry_self_test():
    """Smoke-test: every registry entry must return a working Recipe."""
    for name in list_aggregations():
        agg = make_aggregation(name)
        out = agg.apply(np.array([0.1, 0.5, 0.9]))
        assert np.isfinite(out)
    for name in list_calibrators():
        cal = make_calibrator(name)
        # cv_mix needs list-of-parts input; others take scalar vectors
        if name == "cv_mix":
            cal.fit([np.array([0.1, 0.9]), np.array([0.2, 0.8])],
                    np.array([0.0, 1.0]), k_folds=2)
            out = cal.apply([np.array([0.1, 0.9]), np.array([0.2, 0.8])])
        else:
            if hasattr(cal, "fit"):
                try:
                    cal.fit(np.array([0.1, 0.9]), np.array([0.0, 1.0]))
                except TypeError:
                    pass
            out = cal.apply(np.array([0.5]))
        assert out is not None
