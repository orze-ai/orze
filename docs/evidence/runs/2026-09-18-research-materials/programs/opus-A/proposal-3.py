import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from scipy.optimize import nnls


def _align(A, names_src, names_ref):
    try:
        s = list(names_src)
        r = list(names_ref)
        if s == r:
            return A
        pos = {n: i for i, n in enumerate(s)}
        cols = [pos[n] for n in r]
        return A[:, cols]
    except Exception:
        return A


def _prop_table(C, X, names):
    names = [str(n) for n in names]
    idx = [i for i, n in enumerate(names) if n.startswith('wtd_mean_')]
    if not idx:
        idx = [i for i, n in enumerate(names) if 'wtd_mean' in n]
    if not idx:
        return np.zeros((C.shape[1], 0), dtype=float)
    P = np.zeros((C.shape[1], len(idx)), dtype=float)
    for j, i in enumerate(idx):
        try:
            sol = np.linalg.lstsq(C, X[:, i], rcond=None)[0]
        except Exception:
            sol = np.zeros(C.shape[1], dtype=float)
        P[:, j] = np.nan_to_num(sol, nan=0.0, posinf=0.0, neginf=0.0)
    return P


def _wq(vs, cw, q):
    k = int(np.searchsorted(cw, q))
    if k >= vs.shape[0]:
        k = vs.shape[0] - 1
    if k < 0:
        k = 0
    return float(vs[k])


def _derive(C, P):
    n = C.shape[0]
    k = P.shape[1]
    ns = 7
    D = np.zeros((n, k * ns + 12), dtype=float)
    for r in range(n):
        c = C[r]
        nz = np.nonzero(c > 1e-12)[0]
        if nz.size == 0:
            continue
        w = c[nz].astype(float)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        w = w / s
        for j in range(k):
            v = P[nz, j]
            o = np.argsort(v)
            vs = v[o]
            ws = w[o]
            cw = np.cumsum(ws)
            mu = float(np.dot(w, v))
            var = float(np.dot(w, (v - mu) ** 2))
            if var < 0:
                var = 0.0
            sd = float(np.sqrt(var))
            sk = float(np.dot(w, (v - mu) ** 3) / (sd ** 3 + 1e-12))
            b = j * ns
            D[r, b + 0] = float(vs[0])
            D[r, b + 1] = float(vs[-1])
            D[r, b + 2] = float(vs[-1] - vs[0])
            D[r, b + 3] = _wq(vs, cw, 0.25)
            D[r, b + 4] = _wq(vs, cw, 0.5)
            D[r, b + 5] = _wq(vs, cw, 0.75)
            D[r, b + 6] = sk
        b = k * ns
        sw = np.sort(w)[::-1]
        D[r, b + 0] = float(nz.size)
        D[r, b + 1] = float(sw[0])
        D[r, b + 2] = float(sw[1]) if sw.size > 1 else 0.0
        D[r, b + 3] = float(sw[2]) if sw.size > 2 else 0.0
        D[r, b + 4] = float(sw[-1])
        D[r, b + 5] = float(sw[0] - sw[-1])
        D[r, b + 6] = float(np.dot(w, w))
        D[r, b + 7] = float(-np.dot(w, np.log(w + 1e-12)))
        D[r, b + 8] = float(np.std(w))
        D[r, b + 9] = float(sw[0] / (sw[1] + 1e-9)) if sw.size > 1 else 0.0
        D[r, b + 10] = float(np.sum(w < 0.1))
        D[r, b + 11] = float(np.sum(w > 0.3))
    return D


def _groups(C):
    B = C > 1e-12
    seen = {}
    g = np.zeros(B.shape[0], dtype=int)
    for i in range(B.shape[0]):
        key = B[i].tobytes()
        if key not in seen:
            seen[key] = len(seen)
        g[i] = seen[key]
    return g


def _clean(A):
    return np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _specs(rs):
    return [
        ('full', HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                               max_leaf_nodes=31, min_samples_leaf=15,
                                               l2_regularization=1.0, early_stopping=False,
                                               random_state=rs + 1)),
        ('full', HistGradientBoostingRegressor(max_iter=350, learning_rate=0.08,
                                               max_leaf_nodes=63, min_samples_leaf=20,
                                               l2_regularization=3.0, early_stopping=False,
                                               random_state=rs + 2)),
        ('cd', HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                             max_leaf_nodes=31, min_samples_leaf=15,
                                             l2_regularization=1.0, early_stopping=False,
                                             random_state=rs + 3)),
        ('full', ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2,
                                     max_features=0.3, n_jobs=1, random_state=rs + 4)),
        ('full', make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
    ]


def _run(train, inputs, seed):
    rs = int(seed) if seed is not None else 0
    Xtr = _clean(train['X'])
    Ctr = _clean(train['C'])
    ytr = np.asarray(train['y'], dtype=float)
    fn_t = list(train['feature_names'])
    el_t = list(train['elements'])
    Xe = _clean(_align(_clean(inputs['X']), list(inputs['feature_names']), fn_t))
    Ce = _clean(_align(_clean(inputs['C']), list(inputs['elements']), el_t))
    P = _prop_table(Ctr, Xtr, fn_t)
    Dtr = _clean(_derive(Ctr, P))
    De = _clean(_derive(Ce, P))
    Ftr = np.hstack([Xtr, Ctr, Dtr])
    Fe = np.hstack([Xe, Ce, De])
    CDtr = np.hstack([Ctr, Dtr])
    CDe = np.hstack([Ce, De])
    views = {'full': (Ftr, Fe), 'cd': (CDtr, CDe)}
    g = _groups(Ctr)
    nsp = len(_specs(rs))
    OOF = np.zeros((Ftr.shape[0], nsp), dtype=float)
    ok = True
    try:
        nsplits = 4
        if len(np.unique(g)) < nsplits:
            nsplits = 2
        gkf = GroupKFold(n_splits=nsplits)
        for tr_i, te_i in gkf.split(Ftr, ytr, groups=g):
            ms = _specs(rs)
            for j, (view, est) in enumerate(ms):
                A = views[view][0]
                est.fit(A[tr_i], ytr[tr_i])
                OOF[te_i, j] = est.predict(A[te_i])
    except Exception:
        ok = False
    wts = np.ones(nsp, dtype=float) / float(nsp)
    if ok:
        try:
            cand = nnls(np.nan_to_num(OOF, nan=0.0, posinf=0.0, neginf=0.0), ytr)[0]
            if np.isfinite(cand).all() and float(cand.sum()) > 1e-6:
                wts = cand
        except Exception:
            pass
    preds = np.zeros((Fe.shape[0], nsp), dtype=float)
    finals = _specs(rs)
    for j, (view, est) in enumerate(finals):
        A, B = views[view]
        est.fit(A, ytr)
        preds[:, j] = est.predict(B)
    out = preds.dot(wts)
    med = float(np.median(ytr))
    out = np.nan_to_num(out, nan=med, posinf=med, neginf=med)
    hi = float(np.max(ytr)) * 1.05
    out = np.clip(out, 0.0, hi)
    return [float(v) for v in out]


def fit_predict(train, inputs, seed=0):
    try:
        return _run(train, inputs, seed)
    except Exception:
        Xtr = _clean(train['X'])
        ytr = np.asarray(train['y'], dtype=float)
        Xe = _clean(inputs['X'])
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          random_state=1729)
        m.fit(Xtr, ytr)
        p = m.predict(Xe)
        med = float(np.median(ytr))
        p = np.nan_to_num(p, nan=med, posinf=med, neginf=med)
        return [float(v) for v in np.clip(p, 0.0, float(np.max(ytr)) * 1.05)]
