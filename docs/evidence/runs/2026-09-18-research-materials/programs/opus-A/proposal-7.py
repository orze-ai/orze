import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import nnls


def _comp(C):
    C = np.asarray(C, dtype=np.float64)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.maximum(C, 0.0)
    s = C.sum(axis=1, keepdims=True)
    s[s <= 0.0] = 1.0
    Cn = C / s
    m = Cn > 1e-8
    pres = m.astype(np.float64)
    n_el = pres.sum(axis=1, keepdims=True)
    lg = np.where(m, np.log(np.where(m, Cn, 1.0)), 0.0)
    ent = -(Cn * lg).sum(axis=1, keepdims=True)
    srt = -np.sort(-Cn, axis=1)[:, :4]
    simp = (Cn * Cn).sum(axis=1, keepdims=True)
    gap = srt[:, :1] - srt[:, 1:2]
    extra = np.hstack([n_el, ent, simp, srt, gap])
    return Cn, pres, extra


def _build(X, C):
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Cn, pres, extra = _comp(C)
    FA = np.hstack([X, extra])
    FB = np.hstack([FA, Cn, pres])
    return FA, FB


def _make(kind, s):
    if kind == 'hgb_a':
        return HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06,
                                             max_leaf_nodes=31, min_samples_leaf=15,
                                             l2_regularization=1.0,
                                             early_stopping=False, random_state=s)
    if kind == 'hgb_b':
        return HistGradientBoostingRegressor(max_iter=350, learning_rate=0.05,
                                             max_leaf_nodes=63, min_samples_leaf=25,
                                             l2_regularization=2.0,
                                             early_stopping=False, random_state=s)
    return ExtraTreesRegressor(n_estimators=200, max_features=0.35,
                               min_samples_leaf=3, bootstrap=False,
                               random_state=s, n_jobs=1)


def _fit_one(sp, FA, FB, y, idx, s):
    F = FA if sp['fs'] == 'A' else FB
    t = y[idx]
    if sp['tf'] == 'log':
        t = np.log1p(np.maximum(t, 0.0))
    est = _make(sp['k'], s)
    est.fit(F[idx], t)
    return est


def _pred(sp, est, F):
    p = np.asarray(est.predict(F), dtype=np.float64)
    if sp['tf'] == 'log':
        p = np.expm1(np.clip(p, -5.0, 12.0))
    return np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train, inputs, seed):
    try:
        s0 = int(seed)
    except Exception:
        s0 = 0
    y = np.asarray(train['y'], dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    FA, FB = _build(train['X'], train['C'])
    TA, TB = _build(inputs['X'], inputs['C'])
    n_eval = TA.shape[0]
    med = float(np.median(y)) if y.size else 0.0
    ymax = float(np.max(y)) if y.size else 1.0

    g = train.get('groups', None)
    if g is None:
        gi = np.arange(y.shape[0])
    else:
        gi = np.unique(np.asarray([str(v) for v in g]), return_inverse=True)[1]
    n_groups = int(np.unique(gi).shape[0])

    specs = [
        {'fs': 'B', 'tf': 'id', 'k': 'hgb_a'},
        {'fs': 'B', 'tf': 'log', 'k': 'hgb_a'},
        {'fs': 'A', 'tf': 'id', 'k': 'hgb_b'},
        {'fs': 'B', 'tf': 'id', 'k': 'et'},
    ]

    n_folds = 4 if n_groups >= 8 else 2
    try:
        folds = list(GroupKFold(n_splits=n_folds).split(FA, y, gi))
    except Exception:
        folds = []

    oof_list = []
    full_list = []
    all_idx = np.arange(y.shape[0])
    for k, sp in enumerate(specs):
        try:
            o = np.zeros(y.shape[0], dtype=np.float64)
            if folds:
                for tr_i, va_i in folds:
                    est = _fit_one(sp, FA, FB, y, tr_i, s0 + 11 * k + 1)
                    Fv = (FA if sp['fs'] == 'A' else FB)[va_i]
                    o[va_i] = _pred(sp, est, Fv)
            else:
                o[:] = med
            estf = _fit_one(sp, FA, FB, y, all_idx, s0 + 11 * k + 1)
            pf = _pred(sp, estf, TA if sp['fs'] == 'A' else TB)
        except Exception:
            continue
        if o.shape[0] == y.shape[0] and pf.shape[0] == n_eval:
            if np.all(np.isfinite(o)) and np.all(np.isfinite(pf)):
                oof_list.append(o)
                full_list.append(pf)

    if not oof_list:
        return [med] * n_eval

    P = np.vstack(oof_list).T
    Q = np.vstack(full_list).T
    rm = np.sqrt(((P - y[:, None]) ** 2).mean(axis=0))
    j = int(np.argmin(rm))
    pred = Q[:, j]
    if folds:
        A = np.hstack([P, np.ones((P.shape[0], 1), dtype=np.float64)])
        try:
            w, _ = nnls(A, y)
            if np.all(np.isfinite(w)) and float(w[:-1].sum()) > 1e-8:
                br = float(np.sqrt(((A.dot(w) - y) ** 2).mean()))
                if br <= float(rm.min()) + 1e-12:
                    cand = Q.dot(w[:-1]) + w[-1]
                    if np.all(np.isfinite(cand)):
                        pred = cand
        except Exception:
            pass

    pred = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=med,
                         posinf=ymax, neginf=0.0)
    pred = np.clip(pred, 0.0, ymax * 1.05)
    return [float(v) for v in pred]
