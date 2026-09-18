import numpy as np
from scipy.optimize import least_squares

_DEF = np.array([2.22, 0.36, 1.0, 0.083, 1.0, 0.14, 1.0])
_LB = np.array([0.5, 0.01, 0.3, 0.005, 0.3, 0.0, 0.0])
_UB = np.array([6.0, 5.0, 4.0, 0.5, 3.0, 1.0, 4.0])


def _f(u, A, c, p):
    u = np.clip(u, 0.0, 1.0)
    return A * u ** p / (c ** p + u ** p)


def _sim(u, th):
    A, c, p, k, q, r, m = [float(v) for v in th]
    S = 1.0
    ys = np.empty(len(u))
    for t in range(len(u)):
        ut = min(max(float(u[t]), 0.0), 1.0)
        ys[t] = _f(ut, A, c, p) * S
        a = k * ut ** q
        b = r * (1.0 - ut) ** m
        tot = a + b
        if tot > 1e-12:
            Ss = b / tot
            S = Ss + (S - Ss) * np.exp(-tot)
    return ys


def _resid(th, data):
    return np.concatenate([_sim(u, th) - y for u, y in data])


def _fit(data, seed):
    rng = np.random.default_rng(int(seed) if seed is not None else 0)
    starts = [_DEF.copy()]
    for _ in range(5):
        s = _DEF * np.exp(rng.normal(0.0, 0.3, size=7))
        starts.append(np.clip(s, _LB + 1e-6, _UB - 1e-6))
    best = None
    for s in starts:
        try:
            res = least_squares(_resid, s, bounds=(_LB, _UB), args=(data,), max_nfev=2000)
            if np.all(np.isfinite(res.x)) and (best is None or res.cost < best.cost):
                best = res
        except Exception:
            continue
    return best.x if best is not None else _DEF.copy()


def predict(observations, protocols, seed):
    data = []
    for ob in observations or []:
        try:
            u = np.asarray(ob['protocol']['u'], dtype=float)
            y = np.asarray(ob['y'], dtype=float)
        except Exception:
            continue
        n = min(len(u), len(y))
        if n > 0 and np.all(np.isfinite(y[:n])):
            data.append((u[:n], y[:n]))
    th = _fit(data, seed) if data else _DEF.copy()
    out = []
    for pr in protocols:
        u = np.asarray(pr['u'], dtype=float)
        ys = _sim(u, th)
        ys = np.where(np.isfinite(ys), ys, 0.0)
        out.append([float(v) for v in ys])
    return out
