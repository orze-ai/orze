import math
import numpy as np
from scipy.optimize import least_squares

# Model: y[t] = g(u[t]) * x[t];  x[0]=1;  x[t+1] = x[t]*exp(-k*u[t]**p)
# g(u) = gmax * u**n / (K**n + u**n)   (saturating gain, g(0)=0)
# theta = [gmax, K, n, k, p]
_DEFAULT = np.array([2.237, 0.369, 1.0, 0.082, 1.0])
_LB = np.array([0.3, 0.01, 0.3, 1e-4, 0.3])
_UB = np.array([10.0, 10.0, 4.0, 1.0, 3.0])


def _g(u, gmax, K, n):
    u = max(float(u), 0.0)
    if u <= 0.0:
        return 0.0
    un = u ** n
    return gmax * un / (K ** n + un)


def _simulate(u, theta):
    gmax, K, n, k, p = [float(v) for v in theta]
    out = np.zeros(len(u), dtype=float)
    x = 1.0
    for i, ui in enumerate(u):
        ui = float(ui)
        if not np.isfinite(ui):
            ui = 0.0
        ui = min(max(ui, 0.0), 1.0)
        out[i] = _g(ui, gmax, K, n) * x
        x *= math.exp(-k * (ui ** p))
    return out


def _collect(observations):
    data = []
    for ob in observations or []:
        try:
            u = [float(v) for v in ob['protocol']['u']]
            t = [int(v) for v in ob['t']]
            y = [float(v) for v in ob['y']]
        except Exception:
            continue
        if len(t) != len(y) or len(u) == 0:
            continue
        idx = []
        yy = []
        for ti, yi in zip(t, y):
            if 0 <= ti < len(u) and np.isfinite(yi):
                idx.append(ti)
                yy.append(yi)
        if idx:
            data.append((np.array(u), np.array(idx, dtype=int), np.array(yy)))
    return data


def _residuals(theta, data):
    res = []
    for u, idx, y in data:
        pred = _simulate(u, theta)
        res.append(pred[idx] - y)
    if not res:
        return np.zeros(1)
    return np.concatenate(res)


def _fit(data, seed):
    if not data:
        return _DEFAULT.copy()
    best = None
    rng = np.random.RandomState(int(seed) if seed is not None else 0)
    starts = [_DEFAULT.copy()]
    for _ in range(4):
        s = _DEFAULT * np.exp(rng.normal(0.0, 0.2, size=5))
        starts.append(np.clip(s, _LB * 1.01, _UB * 0.99))
    for s in starts:
        try:
            r = least_squares(_residuals, s, args=(data,), bounds=(_LB, _UB),
                              loss='soft_l1', f_scale=0.03, max_nfev=400)
            cost = float(np.sum(_residuals(r.x, data) ** 2))
            if np.isfinite(cost) and (best is None or cost < best[0]):
                best = (cost, r.x.copy())
        except Exception:
            continue
    if best is None:
        return _DEFAULT.copy()
    return best[1]


def predict(observations, protocols, seed):
    data = _collect(observations)
    theta = _fit(data, seed)
    out = []
    for pr in protocols:
        try:
            u = [float(v) for v in pr['u']]
        except Exception:
            u = []
        pred = _simulate(u, theta) if len(u) else np.zeros(0)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        out.append([float(v) for v in pred])
    return out
