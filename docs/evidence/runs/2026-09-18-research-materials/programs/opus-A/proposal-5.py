import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import nnls

# NOTE ON TRANSDUCTION: element-set membership and per-set feature centroids for the
# evaluation pool are computed from the supplied unlabeled inputs X/C only. No
# evaluation labels, groups, row ids or formulas are used anywhere.

EPS = 1e-12


def _keys_from_frac(f):
    P = f > 1e-8
    out = []
    for i in range(P.shape[0]):
        idx = np.flatnonzero(P[i])
        out.append("-".join([str(int(j)) for j in idx]))
    return np.asarray(out, dtype=object)


def _build(X, C, elements):
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    C = np.clip(np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    s = C.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    f = C / s
    srt = np.sort(f, axis=1)[:, ::-1]
    cnt = (f > 1e-8).sum(axis=1).astype(float)
    lg = np.where(f > 1e-10, np.log(np.maximum(f, 1e-10)), 0.0)
    ent = -(f * lg).sum(axis=1)
    idx = {}
    for i, e in enumerate(list(elements)):
        idx[str(e)] = i

    def g(sym):
        j = idx.get(sym, -1)
        if j < 0 or j >= f.shape[1]:
            return np.zeros(n)
        return f[:, j]

    cu = g("Cu")
    ox = g("O")
    fe = g("Fe")
    pn = g("As") + g("P") + g("Se") + g("Te") + g("S")
    cols = [cnt, srt[:, 0], srt[:, 1], srt[:, 2], srt[:, 3], ent,
            cu, ox, fe, pn, cu * ox, fe * pn,
            (cu > 0).astype(float) * (ox > 0).astype(float),
            (fe > 0).astype(float) * (pn > 0).astype(float),
            g("H"), g("B"), g("C"), g("N"), g("Mg"), g("Nb"), g("Bi"),
            g("Ba"), g("Sr"), g("Ca"), g("Y"), g("La"), g("Hg"), g("Tl"),
            g("Pb"), g("Ni"), g("Ti"), g("V")]
    extra = np.column_stack(cols)
    F = np.hstack([np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), f, extra])
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, f


def _centroids(F, keys):
    uk, inv = np.unique(np.asarray(keys, dtype=object), return_inverse=True)
    m = len(uk)
    cnts = np.bincount(inv, minlength=m).astype(float)
    cnts = np.where(cnts <= 0, 1.0, cnts)
    M = np.zeros((m, F.shape[1]), dtype=float)
    for j in range(F.shape[1]):
        M[:, j] = np.bincount(inv, weights=F[:, j], minlength=m) / cnts
    return uk, inv, M, cnts


def _hgb(seed, lr=0.06, it=350, leaf=48, msl=15):
    return HistGradientBoostingRegressor(max_iter=it, learning_rate=lr,
                                         max_leaf_nodes=leaf,
                                         l2_regularization=1.0,
                                         min_samples_leaf=msl,
                                         early_stopping=False,
                                         random_state=seed)


def _two_stage(Ftr, ytr, ktr, Fev, kev, seed):
    uk, inv, M, cnts = _centroids(Ftr, ktr)
    gy = np.bincount(inv, weights=ytr, minlength=len(uk)) / cnts
    ga = _hgb(seed, lr=0.06, it=250, leaf=16, msl=5)
    ga.fit(M, gy)
    uke, inve, Me, _c = _centroids(Fev, kev)
    gpred = ga.predict(Me)[inve]
    Dtr = np.hstack([Ftr, Ftr - M[inv]])
    gb = _hgb(seed + 1, lr=0.06, it=350, leaf=48, msl=15)
    gb.fit(Dtr, ytr - gy[inv])
    Dev = np.hstack([Fev, Fev - Me[inve]])
    return gpred + gb.predict(Dev)


def _predict_all(Ftr, ytr, ktr, Fev, kev, seed):
    preds = []
    m1 = _hgb(seed)
    m1.fit(Ftr, ytr)
    preds.append(m1.predict(Fev))
    m2 = _hgb(seed + 11)
    m2.fit(Ftr, np.log1p(np.clip(ytr, 0.0, None)))
    preds.append(np.expm1(np.clip(m2.predict(Fev), -5.0, 12.0)))
    preds.append(_two_stage(Ftr, ytr, ktr, Fev, kev, seed + 23))
    m4 = ExtraTreesRegressor(n_estimators=120, max_features=0.35,
                             min_samples_leaf=2, random_state=seed + 37,
                             n_jobs=1)
    m4.fit(Ftr, ytr)
    preds.append(m4.predict(Fev))
    P = np.column_stack(preds)
    return np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train, inputs, seed=0):
    try:
        sd = int(seed)
    except Exception:
        sd = 0
    els = inputs.get("elements")
    if els is None:
        els = train.get("elements")
    if els is None:
        els = []
    Ftr, ftr = _build(train["X"], train["C"], els)
    ytr = np.asarray(train["y"], dtype=float)
    ytr = np.nan_to_num(ytr, nan=0.0, posinf=0.0, neginf=0.0)
    gtr = train.get("groups")
    if gtr is None:
        ktr = _keys_from_frac(ftr)
    else:
        ktr = np.asarray([str(g) for g in gtr], dtype=object)
    Fev, fev = _build(inputs["X"], inputs["C"], els)
    kev = _keys_from_frac(fev)
    nrow_ev = Fev.shape[0]
    ymax = float(np.max(ytr)) if ytr.size else 1.0
    hi = 1.05 * ymax
    med = float(np.median(ytr)) if ytr.size else 0.0

    w = np.array([1.0, 0.0, 0.0, 0.0])
    try:
        ng = len(np.unique(ktr))
        nsp = 3 if ng >= 3 else 2
        oof = np.zeros((Ftr.shape[0], 4), dtype=float)
        seen = np.zeros(Ftr.shape[0], dtype=bool)
        gkf = GroupKFold(n_splits=nsp)
        for tr, te in gkf.split(Ftr, ytr, groups=ktr):
            oof[te] = _predict_all(Ftr[tr], ytr[tr], ktr[tr], Ftr[te], ktr[te], sd)
            seen[te] = True
        if seen.all():
            oof = np.clip(np.nan_to_num(oof, nan=med, posinf=hi, neginf=0.0), 0.0, hi)
            ww, _r = nnls(oof, ytr)
            ww = np.nan_to_num(np.asarray(ww, dtype=float), nan=0.0)
            if np.isfinite(ww).all() and ww.sum() > 1e-6 and ww.sum() < 5.0:
                w = ww
            else:
                rm = [float(np.sqrt(np.mean((oof[:, j] - ytr) ** 2))) for j in range(4)]
                b = int(np.argmin(rm))
                w = np.zeros(4)
                w[b] = 1.0
    except Exception:
        w = np.array([1.0, 0.0, 0.0, 0.0])

    try:
        P = _predict_all(Ftr, ytr, ktr, Fev, kev, sd)
        pred = P.dot(w)
    except Exception:
        mf = _hgb(sd)
        mf.fit(Ftr, ytr)
        pred = mf.predict(Fev)
    pred = np.asarray(pred, dtype=float).ravel()
    if pred.shape[0] != nrow_ev:
        pred = np.full(nrow_ev, med, dtype=float)
    pred = np.nan_to_num(pred, nan=med, posinf=hi, neginf=0.0)
    pred = np.clip(pred, 0.0, hi)
    return [float(v) for v in pred]
