import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR


def _hgb(seed):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=5.0,
        max_iter=100,
        early_stopping=False,
        random_state=seed,
    )


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = np.asarray(X_train, dtype=float)
    Xev = np.asarray(X_eval, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    ymean = float(np.mean(y))
    lo = float(np.min(y)) - 5.0
    hi = float(np.max(y)) + 5.0

    parts = []

    # member 1: raw-target histogram boosting (best observed development config)
    m1 = _hgb(seed)
    m1.fit(Xtr, y)
    parts.append(np.asarray(m1.predict(Xev), dtype=float).ravel())

    # member 2: same learner on log target (variance stabilising for high-MPG rows)
    ly = np.log(np.clip(y, 1e-3, None))
    m2 = _hgb(seed)
    m2.fit(Xtr, ly)
    p2 = np.exp(np.asarray(m2.predict(Xev), dtype=float).ravel())
    parts.append(p2)

    # member 3: smooth kernel model with standardised features and target
    m3 = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=0.1)),
        transformer=StandardScaler(),
    )
    m3.fit(Xtr, y)
    parts.append(np.asarray(m3.predict(Xev), dtype=float).ravel())

    P = np.vstack(parts)
    P = np.where(np.isfinite(P), P, ymean)
    pred = P.mean(axis=0)
    pred = np.where(np.isfinite(pred), pred, ymean)
    pred = np.clip(pred, lo, hi)
    return np.asarray(pred, dtype=float).ravel()
