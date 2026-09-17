import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor


def _feat(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    p = X[:, 10:11]
    v = X[:, 11:12]
    extra = np.hstack([p - v, p * v, p + v, v / (p + 1.0), p / (v + 1.0)])
    return np.hstack([X, extra])


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = _feat(X_train)
    Xev = _feat(X_eval)
    y = np.asarray(y_train, dtype=float).ravel()
    lo = float(np.min(y))
    hi = float(np.max(y))
    ylog = np.log(np.clip(y, 1e-6, None))
    parts = []

    g1 = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.2,
                                       max_leaf_nodes=63, min_samples_leaf=5,
                                       l2_regularization=1.0, max_iter=300,
                                       early_stopping=False, random_state=seed)
    g1.fit(Xtr, y)
    parts.append((0.45, np.asarray(g1.predict(Xev), dtype=float)))

    g2 = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.1,
                                       max_leaf_nodes=31, min_samples_leaf=5,
                                       l2_regularization=1.0, max_iter=400,
                                       early_stopping=False, random_state=seed + 1)
    g2.fit(Xtr, ylog)
    parts.append((0.25, np.exp(np.clip(np.asarray(g2.predict(Xev), dtype=float), -20.0, 20.0))))

    et = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=1, max_features=0.75,
                             random_state=seed, n_jobs=1)
    et.fit(Xtr, ylog)
    parts.append((0.30, np.exp(np.clip(np.asarray(et.predict(Xev), dtype=float), -20.0, 20.0))))

    out = np.zeros(Xev.shape[0], dtype=float)
    wsum = 0.0
    for w, p in parts:
        p = np.where(np.isfinite(p), p, float(np.mean(y)))
        out = out + w * p
        wsum += w
    out = out / wsum
    out = np.clip(out, 0.5 * lo, 1.5 * hi)
    out = np.where(np.isfinite(out), out, float(np.mean(y)))
    return np.asarray(out, dtype=float).ravel()
