import numpy as np
from scipy.optimize import least_squares


def _sim(u, p):
    A, K, n, k, m = [float(v) for v in p]
    u = np.asarray(u, dtype=float).ravel()
    u = np.clip(u, 0.0, 1.0)
    up = np.power(np.maximum(u, 0.0), n)
    g = A * up / (np.power(max(K, 1e-9), n) + up)
    g = np.where(u <= 0.0, 0.0, g)
    d = np.power(np.maximum(u, 0.0), m)
    cum = np.concatenate(([0.0], np.cumsum(d)[:-1])) if u.size > 0 else np.zeros(0)
    return g * np.exp(-k * cum)


def _collect(observations):
    data = []
    if not observations:
        return data
    for o in observations:
        try:
            prot = o.get('protocol', {})
            u = prot.get('u', None)
            y = o.get('y', None)
        except Exception:
            continue
        if u is None or y is None:
            continue
        try:
            ua = np.asarray(u, dtype=float).ravel()
            ya = np.asarray(y, dtype=float).ravel()
        except Exception:
            continue
        nn = int(min(ua.size, ya.size))
        if nn <= 0:
            continue
        ua = ua[:nn]
        ya = ya[:nn]
        if not np.all(np.isfinite(ua)) or not np.all(np.isfinite(ya)):
            good = np.isfinite(ua) & np.isfinite(ya)
            ua = ua[good]
            ya = ya[good]
            if ua.size == 0:
                continue
        data.append((ua, ya))
    return data


def _resid(p, data):
    parts = []
    for u, y in data:
        parts.append(_sim(u, p) - y)
    if not parts:
        return np.zeros(1)
    r = np.concatenate(parts)
    return np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3)


def _fit(data):
    default = np.array([2.26, 0.385, 1.0, 0.0825, 1.0])
    if not data:
        return default
    lo = [0.3, 0.005, 0.3, 1e-4, 0.3]
    hi = [6.0, 5.0, 4.0, 1.0, 3.0]
    starts = [
        [2.26, 0.385, 1.0, 0.0825, 1.0],
        [2.20, 0.350, 1.0, 0.0800, 1.0],
        [1.90, 0.200, 1.3, 0.0900, 0.9],
        [3.00, 0.800, 0.8, 0.0700, 1.1],
        [1.70, 0.100, 1.6, 0.0850, 1.0],
    ]
    best = None
    best_cost = np.inf
    for p0 in starts:
        try:
            res = least_squares(_resid, p0, bounds=(lo, hi), args=(data,),
                                loss='soft_l1', f_scale=0.05, max_nfev=3000)
        except Exception:
            continue
        if np.isfinite(res.cost) and res.cost < best_cost:
            best_cost = float(res.cost)
            best = np.asarray(res.x, dtype=float)
    if best is None or not np.all(np.isfinite(best)):
        return default
    return best


def predict(observations, protocols, seed=0):
    data = _collect(observations)
    p = _fit(data)
    out = []
    if protocols is None:
        return out
    for pr in protocols:
        if isinstance(pr, dict):
            u = pr.get('u', [])
        else:
            u = pr
        try:
            ua = np.asarray(u, dtype=float).ravel()
        except Exception:
            ua = np.zeros(0)
        ua = np.nan_to_num(ua, nan=0.0, posinf=1.0, neginf=0.0)
        if ua.size == 0:
            out.append([])
            continue
        yv = _sim(ua, p)
        yv = np.nan_to_num(yv, nan=0.0, posinf=0.0, neginf=0.0)
        out.append([float(v) for v in yv])
    return out
