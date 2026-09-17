import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

SEED = 1729


def _make(key):
    if key == 'et':
        return ExtraTreesRegressor(n_estimators=400, max_features=0.5,
                                   min_samples_leaf=1, random_state=SEED, n_jobs=1)
    if key == 'rf':
        return RandomForestRegressor(n_estimators=400, max_features=0.5,
                                     min_samples_leaf=1, random_state=SEED, n_jobs=1)
    return GradientBoostingRegressor(learning_rate=0.05, n_estimators=300,
                                     max_depth=3, min_samples_leaf=3,
                                     random_state=SEED)


def _fit_pred(key, use_log, Xtr, ytr, Xte):
    m = _make(key)
    if use_log:
        m.fit(Xtr, np.log1p(np.maximum(ytr, 0.0)))
        p = np.expm1(m.predict(Xte))
    else:
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
    p = np.asarray(p, dtype=float)
    return np.clip(p, 0.0, None)


CANDS = [
    (('et', False),),
    (('et', True),),
    (('rf', False),),
    (('gb', False),),
    (('et', False), ('et', True)),
    (('et', False), ('gb', False)),
    (('et', False), ('rf', False), ('gb', False)),
]


def _blend(spec, Xtr, ytr, Xte):
    preds = [_fit_pred(k, lg, Xtr, ytr, Xte) for (k, lg) in spec]
    return np.mean(np.vstack(preds), axis=0)


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    folds = []
    try:
        if train_groups is not None:
            g = np.asarray(train_groups).ravel()
            nsp = int(min(5, len(np.unique(g))))
            if nsp >= 2:
                folds = list(GroupKFold(n_splits=nsp).split(X, y, g))
        if not folds and X.shape[0] >= 10:
            folds = list(KFold(n_splits=5, shuffle=True,
                              random_state=SEED).split(X))
    except Exception:
        folds = []
    best = CANDS[0]
    if len(folds) >= 2:
        scores = []
        for spec in CANDS:
            try:
                se = 0.0
                cnt = 0
                for tr, te in folds:
                    if len(tr) < 5 or len(te) < 1:
                        continue
                    p = _blend(spec, X[tr], y[tr], X[te])
                    se += float(np.sum((p - y[te]) ** 2))
                    cnt += int(len(te))
                scores.append(se / cnt if cnt > 0 else float('inf'))
            except Exception:
                scores.append(float('inf'))
        if np.isfinite(np.min(scores)):
            best = CANDS[int(np.argmin(scores))]
    try:
        out = _blend(best, X, y, Xe)
    except Exception:
        out = np.full(Xe.shape[0], float(np.mean(y)))
    out = np.asarray(out, dtype=float).ravel()
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = float(np.mean(y))
    return out
