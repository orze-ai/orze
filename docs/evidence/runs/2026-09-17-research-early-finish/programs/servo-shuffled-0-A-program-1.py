import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    if Xe.ndim == 1:
        Xe = Xe.reshape(1, -1)
    n_eval = Xe.shape[0]
    n = X.shape[0]
    if n == 0:
        return np.zeros(n_eval, dtype=float)
    mean = float(np.mean(y))
    fallback = np.full(n_eval, mean, dtype=float)
    if n_eval == 0:
        return np.zeros(0, dtype=float)
    if n < 20:
        return fallback
    if train_groups is not None:
        g = np.asarray(train_groups).ravel()
        if g.shape[0] != n:
            g = np.arange(n)
    else:
        g = np.arange(n)
    uniq = np.unique(g)
    alphas = [10.0, 100.0, 1000.0]
    try:
        if uniq.shape[0] >= 3:
            k = int(min(5, uniq.shape[0]))
            folds = list(GroupKFold(n_splits=k).split(X, y, groups=g))
        else:
            folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X))
    except Exception:
        return fallback
    const_err = 0.0
    ridge_err = dict((a, 0.0) for a in alphas)
    used = 0
    for tr, te in folds:
        if tr.shape[0] < 5 or te.shape[0] < 1:
            continue
        ytr = y[tr]
        yte = y[te]
        m = float(np.mean(ytr))
        const_err += float(np.sum((yte - m) ** 2))
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Ztr = (X[tr] - mu) / sd
        Zte = (X[te] - mu) / sd
        for a in alphas:
            try:
                r = Ridge(alpha=a)
                r.fit(Ztr, ytr)
                p = np.asarray(r.predict(Zte), dtype=float).ravel()
            except Exception:
                p = np.full(te.shape[0], m, dtype=float)
            if not np.all(np.isfinite(p)):
                p = np.full(te.shape[0], m, dtype=float)
            ridge_err[a] += float(np.sum((yte - p) ** 2))
        used += te.shape[0]
    if used < 10 or const_err <= 0.0:
        return fallback
    best_a = min(alphas, key=lambda a: ridge_err[a])
    if ridge_err[best_a] >= 0.95 * const_err:
        return fallback
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    try:
        model = Ridge(alpha=best_a)
        model.fit((X - mu) / sd, y)
        pred = np.asarray(model.predict((Xe - mu) / sd), dtype=float).ravel()
    except Exception:
        return fallback
    if pred.shape[0] != n_eval or not np.all(np.isfinite(pred)):
        return fallback
    lo = float(np.min(y))
    hi = float(np.max(y))
    span = hi - lo
    pred = np.clip(pred, lo - 0.25 * span, hi + 0.25 * span)
    return pred
