import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold


def _make(seed):
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
    ly = np.log(np.clip(y, 1e-3, None))
    n = Xtr.shape[0]
    smear = 1.0
    try:
        oof = np.zeros(n, dtype=float)
        filled = np.zeros(n, dtype=bool)
        if train_groups is not None:
            g = np.asarray(train_groups).ravel()
            k = int(min(5, len(np.unique(g))))
            if k >= 2:
                splits = list(GroupKFold(n_splits=k).split(Xtr, ly, g))
            else:
                splits = []
        else:
            splits = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(Xtr))
        for tr, te in splits:
            m = _make(seed)
            m.fit(Xtr[tr], ly[tr])
            oof[te] = m.predict(Xtr[te])
            filled[te] = True
        if filled.any():
            r = ly[filled] - oof[filled]
            s = float(np.mean(np.exp(r)))
            if np.isfinite(s) and 0.7 < s < 1.3:
                smear = s
    except Exception:
        smear = 1.0
    model = _make(seed)
    model.fit(Xtr, ly)
    pred = np.exp(model.predict(Xev)) * smear
    pred = np.where(np.isfinite(pred), pred, ymean)
    pred = np.clip(pred, 5.0, 55.0)
    return np.asarray(pred, dtype=float).ravel()
