"""Aggregation + calibration registry (F11).

A per-frame probability stream P (shape [n_frames]) collapses to a single
per-clip score via an *aggregation*. A stream of per-clip scores then passes
through a *calibrator* (identity/Platt/isotonic/etc.) before feeding the
evaluation metric.

Both families are registered in the same module-level REGISTRY dict so
``agg_search`` can iterate ``REGISTRY['aggregations']`` × ``REGISTRY['calibrators']``
uniformly. Each entry exposes:

    .apply(probs: np.ndarray) -> float          # aggregations
    .apply(scores: np.ndarray) -> np.ndarray    # calibrators
    .fit(val_probs, val_labels) -> self         # no-op for deterministic
                                                #   aggregations; required
                                                #   for calibrators.

This module deliberately does NOT touch the test set — all fitting is
delegated to the caller which is responsible for passing val-only data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------- #
# Registry plumbing                                                      #
# ---------------------------------------------------------------------- #


REGISTRY: Dict[str, Dict[str, Callable[..., "Recipe"]]] = {
    "aggregations": {},
    "calibrators": {},
}


def _register(family: str, name: str):
    def deco(factory):
        REGISTRY[family][name] = factory
        factory.__recipe_name__ = name
        factory.__recipe_family__ = family
        return factory
    return deco


class Recipe:
    """Base class for aggregations and calibrators."""
    name: str = "?"
    is_fit: bool = True  # overrides to False when fit() is required

    def fit(self, val_probs, val_labels):  # pragma: no cover - default no-op
        self.is_fit = True
        return self

    def apply(self, x):  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# Aggregations (per-frame probs -> per-clip score)                       #
# ---------------------------------------------------------------------- #


@dataclass
class _DeterministicAgg(Recipe):
    fn: Callable[[np.ndarray], float] = None
    name: str = "?"
    params: Dict[str, Any] = field(default_factory=dict)
    is_fit: bool = True

    def apply(self, probs: np.ndarray) -> float:
        p = np.asarray(probs, dtype=np.float64).ravel()
        if p.size == 0:
            return float("nan")
        return float(self.fn(p))


def _make_det(name: str, fn):
    def factory(**_):
        return _DeterministicAgg(fn=fn, name=name)
    factory.__recipe_name__ = name
    return factory


REGISTRY["aggregations"]["last"] = _make_det("last", lambda p: p[-1])
REGISTRY["aggregations"]["mean"] = _make_det("mean", lambda p: p.mean())
REGISTRY["aggregations"]["max"] = _make_det("max", lambda p: p.max())


def _late_k(k: int):
    def fn(p):
        return float(p[-min(k, p.size):].max())
    return fn


for _k in range(1, 6):
    REGISTRY["aggregations"][f"late_k{_k}"] = _make_det(f"late_k{_k}", _late_k(_k))


@_register("aggregations", "top_k_mean")
def _top_k_mean(k: int = 6, **_):
    def fn(p):
        kk = min(k, p.size)
        return float(np.sort(p)[-kk:].mean())
    return _DeterministicAgg(fn=fn, name=f"top{k}_mean", params={"k": k})


@_register("aggregations", "exp_weighted")
def _exp_weighted(alpha: float = 0.5, **_):
    def fn(p):
        # recency-weighted mean: w_i = alpha^(n-1-i)
        n = p.size
        w = np.array([alpha ** (n - 1 - i) for i in range(n)], dtype=np.float64)
        return float((p * w).sum() / w.sum())
    return _DeterministicAgg(fn=fn, name=f"exp_w({alpha})", params={"alpha": alpha})


@_register("aggregations", "noisy_or")
def _noisy_or(**_):
    def fn(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(1.0 - np.prod(1.0 - p))
    return _DeterministicAgg(fn=fn, name="noisy_or")


@_register("aggregations", "dense_late_softmax")
def _dense_late_softmax(t: float = 1.0, **_):
    def fn(p):
        # temperature-scaled softmax-weighted mean favoring high values near end
        n = p.size
        pos = np.arange(n, dtype=np.float64)
        logits = p / max(t, 1e-6) + 0.25 * (pos / max(n - 1, 1))
        m = logits.max()
        w = np.exp(logits - m)
        w /= w.sum()
        return float((p * w).sum())
    return _DeterministicAgg(fn=fn, name=f"dense_late_softmax(t={t})",
                             params={"t": t})


# ---------------------------------------------------------------------- #
# Calibrators (per-clip scores -> per-clip calibrated scores)            #
# ---------------------------------------------------------------------- #


@dataclass
class _Identity(Recipe):
    name: str = "identity"
    is_fit: bool = True

    def fit(self, val_probs, val_labels):
        return self

    def apply(self, scores):
        return np.asarray(scores, dtype=np.float64)


@_register("calibrators", "identity")
def _mk_identity(**_):
    return _Identity()


@dataclass
class _Platt(Recipe):
    """Single-parameter logistic calibrator: sigmoid(a*x + b)."""
    name: str = "platt"
    a: float = 1.0
    b: float = 0.0
    is_fit: bool = False

    def fit(self, val_scores, val_labels, n_iter: int = 200, lr: float = 0.1):
        x = np.asarray(val_scores, dtype=np.float64)
        y = np.asarray(val_labels, dtype=np.float64)
        a, b = 1.0, 0.0
        # small Newton-style / gradient on BCE
        for _ in range(n_iter):
            z = a * x + b
            p = 1.0 / (1.0 + np.exp(-z))
            ga = ((p - y) * x).mean()
            gb = (p - y).mean()
            a -= lr * ga
            b -= lr * gb
        self.a, self.b, self.is_fit = float(a), float(b), True
        return self

    def apply(self, scores):
        x = np.asarray(scores, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-(self.a * x + self.b)))


@_register("calibrators", "platt")
def _mk_platt(**_):
    return _Platt()


@dataclass
class _Isotonic(Recipe):
    """Pool-adjacent-violators isotonic regression."""
    name: str = "isotonic"
    xs: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    ys: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    is_fit: bool = False

    def fit(self, val_scores, val_labels):
        x = np.asarray(val_scores, dtype=np.float64)
        y = np.asarray(val_labels, dtype=np.float64)
        order = np.argsort(x, kind="stable")
        xs = x[order]
        ys = y[order].copy()
        w = np.ones_like(ys)
        # PAV
        i = 0
        while i < len(ys) - 1:
            if ys[i] > ys[i + 1]:
                new_w = w[i] + w[i + 1]
                new_y = (ys[i] * w[i] + ys[i + 1] * w[i + 1]) / new_w
                ys[i] = new_y
                w[i] = new_w
                # delete i+1 by shifting
                ys = np.delete(ys, i + 1)
                xs = np.delete(xs, i + 1)
                w = np.delete(w, i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1
        self.xs = xs
        self.ys = ys
        self.is_fit = True
        return self

    def apply(self, scores):
        x = np.asarray(scores, dtype=np.float64)
        return np.interp(x, self.xs, self.ys,
                         left=self.ys[0], right=self.ys[-1])


@_register("calibrators", "isotonic")
def _mk_isotonic(**_):
    return _Isotonic()


@dataclass
class _GroupCalibrated(Recipe):
    """Per-group rank-normalization calibrator.

    Groups are passed to ``.apply(scores, groups=...)`` as an int array;
    within each group scores are replaced by their rank/(n_group-1).
    """
    name: str = "group_calibrated"
    is_fit: bool = True

    def fit(self, val_probs, val_labels, **_):
        return self

    def apply(self, scores, groups=None):
        x = np.asarray(scores, dtype=np.float64)
        if groups is None:
            return _rank_normalize(x)
        g = np.asarray(groups)
        out = np.zeros_like(x)
        for gid in np.unique(g):
            mask = g == gid
            out[mask] = _rank_normalize(x[mask])
        return out


@_register("calibrators", "group_calibrated")
def _mk_group(**_):
    return _GroupCalibrated()


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    if x.size <= 1:
        return np.zeros_like(x)
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return order.astype(np.float64) / (x.size - 1)


@dataclass
class _CVMix(Recipe):
    """CV-mix: alpha * part_a + (1 - alpha) * part_b.

    ``parts`` is a list of (name, callable) pairs producing per-clip score
    arrays. ``alpha`` is fit on val via a simple 1-D grid with K-fold CV.
    """
    name: str = "cv_mix"
    parts: List[tuple] = field(default_factory=list)
    alpha: float = 0.5
    is_fit: bool = False

    def fit(self, val_parts_scores: Sequence[np.ndarray],
            val_labels: np.ndarray,
            k_folds: int = 5,
            metric: Optional[Callable] = None):
        """val_parts_scores: list of arrays [a_scores, b_scores], one per part."""
        if len(val_parts_scores) != 2:
            raise ValueError("cv_mix currently supports exactly 2 parts")
        a = np.asarray(val_parts_scores[0], dtype=np.float64)
        b = np.asarray(val_parts_scores[1], dtype=np.float64)
        y = np.asarray(val_labels, dtype=np.float64)
        metric = metric or _auroc_like
        grid = np.linspace(0.0, 1.0, 21)
        n = len(y)
        rng = np.random.default_rng(0)
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, k_folds)
        best_alpha, best_score = 0.5, -np.inf
        for alpha in grid:
            fold_scores = []
            for val_fold in folds:
                mask = np.zeros(n, dtype=bool)
                mask[val_fold] = True
                s = alpha * a[mask] + (1 - alpha) * b[mask]
                fold_scores.append(metric(s, y[mask]))
            mean = float(np.mean(fold_scores))
            if mean > best_score:
                best_score = mean
                best_alpha = float(alpha)
        self.alpha = best_alpha
        self.is_fit = True
        return self

    def apply(self, parts_scores):
        a = np.asarray(parts_scores[0], dtype=np.float64)
        b = np.asarray(parts_scores[1], dtype=np.float64)
        return self.alpha * a + (1 - self.alpha) * b


@_register("calibrators", "cv_mix")
def _mk_cv_mix(parts=None, **_):
    return _CVMix(parts=parts or [])


def _auroc_like(scores: np.ndarray, labels: np.ndarray) -> float:
    """Ranking-style metric (AUROC), safe on edge cases."""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Mann-Whitney U
    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    r_pos = ranks[labels > 0.5].sum()
    auc = (r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return float(auc)


# ---------------------------------------------------------------------- #
# Factory helpers                                                        #
# ---------------------------------------------------------------------- #


def make_aggregation(name: str, **kwargs) -> Recipe:
    if name not in REGISTRY["aggregations"]:
        raise KeyError(f"unknown aggregation {name!r}. "
                       f"Known: {sorted(REGISTRY['aggregations'])}")
    return REGISTRY["aggregations"][name](**kwargs)


def make_calibrator(name: str, **kwargs) -> Recipe:
    if name not in REGISTRY["calibrators"]:
        raise KeyError(f"unknown calibrator {name!r}. "
                       f"Known: {sorted(REGISTRY['calibrators'])}")
    return REGISTRY["calibrators"][name](**kwargs)


def list_aggregations() -> List[str]:
    return sorted(REGISTRY["aggregations"])


def list_calibrators() -> List[str]:
    return sorted(REGISTRY["calibrators"])
