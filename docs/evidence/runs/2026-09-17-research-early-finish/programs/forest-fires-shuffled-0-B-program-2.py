import numpy as np
from sklearn.model_selection import KFold, GroupKFold


def _consts(y):
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0:
        return [0.0] * 13
    m = float(np.mean(y))
    out = [m]
    for q in (0.99, 0.975, 0.95, 0.90, 0.85):
        cap = float(np.quantile(y, q))
        out.append(float(np.mean(np.clip(y, 0.0, cap))))
    for a in (0.8, 0.6, 0.45, 0.3, 0.2, 0.12):
        out.append(a * m)
    out.append(float(np.median(y)))
    return out


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    y = np.asarray(y_train, dtype=float).ravel()
    n = int(y.shape[0])
    m_eval = int(np.asarray(X_eval, dtype=float).shape[0])
    full = _consts(y)
    k = len(full)
    c = float(full[0]) if k > 0 else 0.0
    if n >= 20:
        dummy = np.zeros((n, 1), dtype=float)
        folds = []
        if train_groups is not None:
            g = np.asarray(train_groups).ravel()
            if g.shape[0] == n:
                ng = int(np.unique(g).size)
                if ng >= 3:
                    nf = int(min(5, ng))
                    folds = list(GroupKFold(n_splits=nf).split(dummy, y, g))
        if not folds:
            nf = int(min(5, n))
            if nf >= 2:
                for r in range(3):
                    folds = folds + list(KFold(n_splits=nf, shuffle=True, random_state=int(seed) + r).split(dummy))
        totals = np.zeros(k, dtype=float)
        counts = 0.0
        for tr, te in folds:
            if tr.size < 5 or te.size == 0:
                continue
            cs = _consts(y[tr])
            yt = y[te]
            ok = True
            for v in cs:
                if not np.isfinite(v):
                    ok = False
                    break
            if not ok:
                continue
            for i in range(k):
                d = yt - cs[i]
                totals[i] += float(np.sum(d * d))
            counts += float(te.size)
        if counts > 0.0:
            scores = totals / counts
            order = np.argsort(scores)
            top = [full[int(i)] for i in order[:3]]
            vals = [float(v) for v in top if np.isfinite(v)]
            if len(vals) > 0:
                c = float(np.mean(vals))
    if not np.isfinite(c):
        c = float(np.mean(y)) if n > 0 else 0.0
    if not np.isfinite(c):
        c = 0.0
    if c < 0.0:
        c = 0.0
    return np.full(m_eval, float(c), dtype=float)
