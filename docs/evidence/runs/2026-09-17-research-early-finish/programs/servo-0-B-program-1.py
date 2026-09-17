import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


def _feat(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    p = X[:, 10]
    v = X[:, 11]
    eps = 1e-9
    extra = np.column_stack([
        p * v,
        p - v,
        p / (v + eps),
        v / (p + eps),
        np.log(p + 1.0),
        np.log(v + 1.0),
        1.0 / (p + eps),
        1.0 / (v + eps),
        1.0 / (p * v + eps),
    ])
    return np.hstack([X, extra])


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = _feat(X_train)
    Xev = _feat(X_eval)
    y = np.asarray(y_train, dtype=float).ravel()
    fallback = float(np.mean(y)) if y.size else 0.0
    ylog = np.log(np.maximum(y, 1e-6))
    m1 = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=1,
                             max_features=0.6, random_state=seed, n_jobs=1)
    m1.fit(Xtr, ylog)
    m2 = HistGradientBoostingRegressor(learning_rate=0.08, max_leaf_nodes=15,
                                       min_samples_leaf=3, l2_regularization=1.0,
                                       max_iter=400, early_stopping=False,
                                       random_state=seed)
    m2.fit(Xtr, ylog)
    plog = 0.5 * np.asarray(m1.predict(Xev), dtype=float) + 0.5 * np.asarray(m2.predict(Xev), dtype=float)
    plog = np.clip(plog, -12.0, 5.0)
    out = np.exp(plog)
    out = np.where(np.isfinite(out), out, fallback)
    return np.clip(out, 0.0, 50.0).ravel()
