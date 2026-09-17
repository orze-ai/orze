import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor


def _feat(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    p = X[:, 10]
    v = X[:, 11]
    e = 1e-9
    extra = np.column_stack([
        p * v,
        p - v,
        p + v,
        p / (v + e),
        v / (p + e),
        1.0 / (p + e),
        1.0 / (v + e),
        1.0 / (p * v + e),
        np.log(p + 1.0),
        np.log(v + 1.0),
    ])
    return np.hstack([X, extra])


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = _feat(X_train)
    Xev = _feat(X_eval)
    y = np.asarray(y_train, dtype=float).ravel()
    mean = float(np.mean(y)) if y.size else 0.0
    lo = float(np.min(y)) if y.size else 0.0
    hi = float(np.max(y)) if y.size else 1.0
    preds = []
    weights = []
    m1 = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.15,
                                       max_leaf_nodes=31, min_samples_leaf=4,
                                       l2_regularization=1.0, max_iter=400,
                                       early_stopping=False, random_state=seed)
    m1.fit(Xtr, y)
    preds.append(m1.predict(Xev))
    weights.append(0.40)
    m2 = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, max_features=1.0,
                            random_state=seed, n_jobs=1)
    m2.fit(Xtr, y)
    preds.append(m2.predict(Xev))
    weights.append(0.35)
    m3 = RandomForestRegressor(n_estimators=300, min_samples_leaf=1, max_features=0.6,
                              random_state=seed, n_jobs=1)
    m3.fit(Xtr, y)
    preds.append(m3.predict(Xev))
    weights.append(0.25)
    P = np.vstack([np.asarray(q, dtype=float).ravel() for q in preds])
    w = np.asarray(weights, dtype=float)
    w = w / float(np.sum(w))
    out = np.dot(w, P)
    out = np.where(np.isfinite(out), out, mean)
    return np.clip(out, max(0.0, 0.5 * lo), 1.2 * hi).ravel()
