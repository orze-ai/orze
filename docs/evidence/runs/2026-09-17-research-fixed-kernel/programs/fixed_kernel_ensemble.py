"""Fixed SVR/tree ensemble for measured comparisons, without internal tuning.

Uses the same estimators, weights and train-derived output bounds as the
original generated program. ``train_groups`` keeps the evaluator signature;
this variant performs no internal validation. It is an optional candidate,
not a universally preferred model or a replacement for final confirmation.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).ravel()
    Xev = np.asarray(X_eval, dtype=float)
    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    Zev = scaler.transform(Xev)
    svr = SVR(C=1.0, gamma="scale", epsilon=0.1).fit(Ztr, ytr)
    p_svr = svr.predict(Zev)
    et = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=1,
                             max_features=0.8, random_state=int(seed), n_jobs=1)
    et.fit(Xtr, ytr)
    p_et = et.predict(Xev)
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                       max_leaf_nodes=15, min_samples_leaf=10,
                                       l2_regularization=1.0, early_stopping=False,
                                       random_state=int(seed))
    gb.fit(Xtr, ytr)
    p_gb = gb.predict(Xev)
    pred = 0.5 * p_svr + 0.25 * p_et + 0.25 * p_gb
    mu = float(np.mean(ytr))
    pred = np.where(np.isfinite(pred), pred, mu)
    lo = float(np.min(ytr)) - 1.0
    hi = float(np.max(ytr)) + 1.0
    pred = np.clip(pred, lo, hi)
    return np.asarray(pred, dtype=float).ravel()
