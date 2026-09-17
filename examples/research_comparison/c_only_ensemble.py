"""SVR/tree comparison candidate retaining C tuning and fixing gamma to scale.

Four C values use the original five train-group folds: 20 validation fits,
rather than the original 80. All estimators, weights and output bounds match
the generated source. This candidate requires task-level comparison; it does
not choose a final model or establish a universal preference.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).ravel()
    Xev = np.asarray(X_eval, dtype=float)
    n = Xtr.shape[0]
    if train_groups is None:
        groups = np.arange(n)
    else:
        g = np.asarray(train_groups)
        _, groups = np.unique(g, return_inverse=True)
    n_splits = 5
    if np.unique(groups).size >= n_splits and n >= 2 * n_splits:
        cv = list(GroupKFold(n_splits=n_splits).split(Xtr, ytr, groups))
    else:
        k = max(2, min(n_splits, n))
        cv = list(KFold(n_splits=k, shuffle=True, random_state=int(seed)).split(Xtr))
    best = None
    for C in (1.0, 3.0, 10.0, 30.0):
        gam = 'scale'
        errs = []
        for tr, va in cv:
            if tr.size < 5 or va.size < 1:
                continue
            sc = StandardScaler().fit(Xtr[tr])
            m = SVR(C=C, gamma=gam, epsilon=0.1)
            m.fit(sc.transform(Xtr[tr]), ytr[tr])
            p = m.predict(sc.transform(Xtr[va]))
            errs.append(float(np.mean((p - ytr[va]) ** 2)))
        if not errs:
            continue
        e = float(np.mean(errs))
        if best is None or e < best[0]:
            best = (e, C, gam)
    if best is None:
        best = (0.0, 1.0, 'scale')
    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    Zev = scaler.transform(Xev)
    svr = SVR(C=best[1], gamma=best[2], epsilon=0.1).fit(Ztr, ytr)
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
