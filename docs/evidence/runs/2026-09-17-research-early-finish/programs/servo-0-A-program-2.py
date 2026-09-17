import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold


def _feat(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    p = X[:, 10:11]
    v = X[:, 11:12]
    codes = np.arange(5.0).reshape(-1, 1)
    motor = X[:, 0:5].dot(codes)
    screw = X[:, 5:10].dot(codes)
    extra = np.hstack([p * v, p + v, p - v, p / (v + 1.0), v / (p + 1.0),
                       1.0 / (p * v + 1.0), motor, screw])
    return np.hstack([X, extra])


def _fit_pred(Xtr, ytr, Xev, seed):
    ytr = np.asarray(ytr, dtype=float).ravel()
    b = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.2,
                                      max_leaf_nodes=63, min_samples_leaf=5,
                                      l2_regularization=1.0, max_iter=400,
                                      early_stopping=False, random_state=seed)
    b.fit(Xtr, ytr)
    pb = np.asarray(b.predict(Xev), dtype=float)
    ylog = np.log(np.clip(ytr, 1e-6, None))
    e = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, max_features=1.0,
                            random_state=seed, n_jobs=1)
    e.fit(Xtr, ylog)
    pe = np.exp(np.clip(np.asarray(e.predict(Xev), dtype=float), -20.0, 20.0))
    return pb, pe


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = _feat(X_train)
    Xev = _feat(X_eval)
    y = np.asarray(y_train, dtype=float).ravel()
    n = Xtr.shape[0]
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_w = 0.0
    try:
        if train_groups is not None:
            g = np.asarray(train_groups).ravel()
            if g.shape[0] != n:
                g = np.arange(n)
        else:
            g = np.arange(n)
        uniq = np.unique(g)
        k = int(min(5, uniq.shape[0]))
        if k >= 2:
            errs = np.zeros(len(grid), dtype=float)
            for tr, te in GroupKFold(n_splits=k).split(Xtr, y, groups=g):
                if tr.shape[0] < 5 or te.shape[0] < 1:
                    continue
                pb, pe = _fit_pred(Xtr[tr], y[tr], Xtr[te], seed)
                for i, w in enumerate(grid):
                    pr = (1.0 - w) * pb + w * pe
                    pr = np.where(np.isfinite(pr), pr, float(np.mean(y[tr])))
                    errs[i] += float(np.sum((pr - y[te]) ** 2))
            if np.all(np.isfinite(errs)) and np.sum(errs) > 0.0:
                best_w = grid[int(np.argmin(errs))]
    except Exception:
        best_w = 0.0
    pb, pe = _fit_pred(Xtr, y, Xev, seed)
    out = (1.0 - best_w) * pb + best_w * pe
    mean_y = float(np.mean(y))
    lo = float(np.min(y))
    hi = float(np.max(y))
    out = np.where(np.isfinite(out), out, mean_y)
    out = np.clip(out, max(0.0, 0.5 * lo), 1.5 * hi)
    out = np.asarray(out, dtype=float).ravel()
    out = np.where(np.isfinite(out), out, mean_y)
    return out
