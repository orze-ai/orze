import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, KFold

MEMBERS = [(13.889906, 0.635832), (12.465816, 0.8768089)]
WEIGHTS = [1.0, 0.85, 0.7, 0.5]


def _fit_members(Xtr, ytr, Xev, n_eval, fallback):
    acc = np.zeros(n_eval, dtype=float)
    tot = 0.0
    for C, g in MEMBERS:
        try:
            m = SVR(kernel="rbf", C=C, gamma=g, epsilon=0.1, tol=1e-3, cache_size=200, max_iter=-1)
            m.fit(Xtr, ytr)
            p = np.asarray(m.predict(Xev), dtype=float).ravel()
        except Exception:
            continue
        if p.shape[0] != n_eval:
            continue
        p = np.where(np.isfinite(p), p, fallback)
        acc = acc + p
        tot += 1.0
    if tot <= 0.0:
        return None
    return acc / tot


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).ravel()
    Xev = np.asarray(X_eval, dtype=float)
    if Xev.ndim == 1:
        Xev = Xev.reshape(1, -1)
    n_eval = Xev.shape[0]
    mean_all = float(np.mean(ytr)) if ytr.size else 0.0
    if not np.isfinite(mean_all):
        mean_all = 0.0
    best_w = 0.85
    try:
        n = Xtr.shape[0]
        if n >= 20:
            splitter = None
            if train_groups is not None:
                grp = np.asarray(train_groups).ravel()
                if grp.shape[0] == n:
                    uniq = np.unique(grp)
                    if uniq.size >= 3:
                        n_splits = int(min(5, uniq.size))
                        splitter = GroupKFold(n_splits=n_splits).split(Xtr, ytr, groups=grp)
            if splitter is None:
                splitter = KFold(n_splits=5, shuffle=True, random_state=int(seed)).split(Xtr)
            errs = dict((w, 0.0) for w in WEIGHTS)
            cnt = 0
            for tr, va in splitter:
                if tr.size < 10 or va.size < 1:
                    continue
                mu = float(np.mean(ytr[tr]))
                if not np.isfinite(mu):
                    mu = 0.0
                p = _fit_members(Xtr[tr], ytr[tr], Xtr[va], int(va.size), mu)
                if p is None:
                    continue
                for w in WEIGHTS:
                    q = np.clip(w * p + (1.0 - w) * mu, 0.0, None)
                    errs[w] += float(np.sum((q - ytr[va]) ** 2))
                cnt += int(va.size)
            if cnt > 0:
                best_w = min(WEIGHTS, key=lambda w: (errs[w], -w))
    except Exception:
        best_w = 0.85
    p_full = _fit_members(Xtr, ytr, Xev, n_eval, mean_all)
    if p_full is None:
        out = np.full(n_eval, mean_all, dtype=float)
    else:
        out = best_w * p_full + (1.0 - best_w) * mean_all
    out = np.where(np.isfinite(out), out, mean_all)
    out = np.clip(np.asarray(out, dtype=float).ravel(), 0.0, None)
    return out
