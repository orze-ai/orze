import math
import numpy as np
from scipy.optimize import least_squares

SIGMA = 0.015

# parameters: A, K, n, d0, r0, pe, qe, S0
LO = np.array([0.5, 0.02, 0.4, 1e-4, 1e-4, 0.3, 0.3, 0.5])
HI = np.array([5.0, 3.00, 3.0, 1.0, 1.0, 3.0, 3.0, 1.6])
STARTS = [
    [2.12, 0.34, 1.0, 0.082, 0.152, 1.0, 1.0, 1.03],
    [1.90, 0.20, 1.0, 0.090, 0.130, 1.0, 1.0, 1.00],
    [2.60, 0.60, 1.2, 0.070, 0.180, 1.1, 0.9, 1.00],
    [1.70, 0.10, 0.8, 0.100, 0.200, 0.9, 1.1, 1.00],
]
FALLBACK = np.array([2.12, 0.34, 1.0, 0.082, 0.152, 1.0, 1.0, 1.03])


def _readout(x, A, K, n):
    if x <= 0.0:
        return 0.0
    un = x ** n
    kn = K ** n
    d = un + kn
    if d <= 1e-12:
        return 0.0
    return A * un / d


def _simulate(u, p):
    A, K, n, d0, r0, pe, qe, S0 = [float(v) for v in p]
    T = len(u)
    out = np.empty(T, dtype=float)
    S = S0
    for t in range(T):
        x = float(u[t])
        if not np.isfinite(x):
            x = 0.0
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        out[t] = S * _readout(x, A, K, n)
        a = d0 * (x ** pe) if x > 0.0 else 0.0
        v = 1.0 - x
        b = r0 * (v ** qe) if v > 0.0 else 0.0
        k = a + b
        if k > 1e-12:
            Sinf = b / k
            S = Sinf + (S - Sinf) * math.exp(-k)
    return out


def _dataset(observations):
    data = []
    for rec in (observations or []):
        try:
            u = [float(v) for v in rec['protocol']['u']]
            tt = [int(v) for v in rec['t']]
            yy = [float(v) for v in rec['y']]
        except Exception:
            continue
        if not u or not tt or len(tt) != len(yy):
            continue
        idx = [i for i in range(len(tt))
               if 0 <= tt[i] < len(u) and np.isfinite(yy[i])]
        if not idx:
            continue
        data.append((u,
                     np.array([tt[i] for i in idx], dtype=int),
                     np.array([yy[i] for i in idx], dtype=float)))
    return data


def _resid(p, data):
    chunks = []
    for u, tt, yy in data:
        m = _simulate(u, p)
        chunks.append((yy - m[tt]) / SIGMA)
    if not chunks:
        return np.zeros(1, dtype=float)
    r = np.concatenate(chunks)
    bad = ~np.isfinite(r)
    if bad.any():
        r[bad] = 1.0e6
    return r


def _fit(data):
    best = None
    bestc = np.inf
    for s in STARTS:
        p0 = np.clip(np.asarray(s, dtype=float), LO, HI)
        try:
            res = least_squares(_resid, p0, args=(data,), bounds=(LO, HI),
                                x_scale='jac', max_nfev=4000)
        except Exception:
            continue
        try:
            c = float(np.sum(np.asarray(res.fun, dtype=float) ** 2))
        except Exception:
            continue
        if np.isfinite(c) and c < bestc:
            bestc = c
            best = np.clip(np.asarray(res.x, dtype=float), LO, HI)
    if best is None:
        return FALLBACK.copy()
    return best


def predict(observations, protocols, seed=0):
    try:
        data = _dataset(observations)
    except Exception:
        data = []
    if data:
        try:
            p = _fit(data)
        except Exception:
            p = FALLBACK.copy()
    else:
        p = FALLBACK.copy()
    outs = []
    for pr in (protocols or []):
        u = None
        try:
            u = [float(v) for v in pr['u']]
        except Exception:
            try:
                u = [float(v) for v in pr]
            except Exception:
                u = None
        if not u:
            outs.append([])
            continue
        try:
            yhat = np.asarray(_simulate(u, p), dtype=float)
        except Exception:
            yhat = np.asarray(_simulate(u, FALLBACK), dtype=float)
        bad = ~np.isfinite(yhat)
        if bad.any():
            yhat[bad] = 0.0
        yhat = np.clip(yhat, 0.0, 5.0)
        outs.append([float(v) for v in yhat])
    return outs
