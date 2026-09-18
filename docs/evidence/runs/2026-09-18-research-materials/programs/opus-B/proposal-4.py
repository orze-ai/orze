import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

TOL = 1e-9


def _keys(C):
    B = np.asarray(C, dtype=float) > TOL
    return np.array([B[i].tobytes() for i in range(B.shape[0])])


def _agg(F, inv, ng):
    d = F.shape[1]
    S = np.zeros((ng, d), dtype=float)
    for j in range(d):
        S[:, j] = np.bincount(inv, weights=F[:, j], minlength=ng)
    cnt = np.bincount(inv, minlength=ng).astype(float)
    cnt[cnt == 0] = 1.0
    return S / cnt[:, None], cnt


def _hgb(seed, mi=250, lr=0.07):
    return HistGradientBoostingRegressor(
        max_iter=mi, learning_rate=lr, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=int(seed) % 2147483647)


def _core(Xa, Ca, ya, Xb, Cb, seed):
    Fa = np.hstack([Xa, Ca])
    Fb = np.hstack([Xb, Cb])
    nX = Xa.shape[1]
    ka = _keys(Ca)
    kb = _keys(Cb)
    ua, ia = np.unique(ka, return_inverse=True)
    ub, ib = np.unique(kb, return_inverse=True)
    GMa, ca = _agg(Fa, ia, len(ua))
    GMb, cb = _agg(Fb, ib, len(ub))
    Pa = (GMa[:, nX:] > TOL).astype(float)
    Pb = (GMb[:, nX:] > TOL).astype(float)
    gy = np.bincount(ia, weights=ya, minlength=len(ua)) / ca
    gm = _hgb(seed, mi=400, lr=0.06)
    gm.fit(np.hstack([GMa, Pa]), gy, sample_weight=ca)
    gpb = gm.predict(np.hstack([GMb, Pb]))
    dm = _hgb(seed + 1)
    dm.fit(np.hstack([Fa - GMa[ia], Pa[ia]]), ya - gy[ia])
    dpb = dm.predict(np.hstack([Fb - GMb[ib], Pb[ib]]))
    two = gpb[ib] + dpb
    fm = _hgb(seed + 2)
    fm.fit(Fa, ya)
    flat = fm.predict(Fb)
    return two, flat, cb[ib], len(ua), len(ub)


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))


def fit_predict(train, inputs, seed=0):
    Xtr = np.asarray(train["X"], dtype=float)
    Ctr = np.asarray(train["C"], dtype=float)
    ytr = np.asarray(train["y"], dtype=float)
    Xev = np.asarray(inputs["X"], dtype=float)
    Cev = np.asarray(inputs["C"], dtype=float)
    els = [str(e) for e in (train.get("elements") or [])]
    keys = _keys(Ctr)
    n = ytr.shape[0]
    oof_two = np.zeros(n)
    oof_flat = np.zeros(n)
    oof_gsize = np.ones(n)
    nsp = 4
    gkf = GroupKFold(n_splits=nsp)
    for f, (tr, va) in enumerate(gkf.split(Xtr, ytr, groups=keys)):
        two, flat, gsz, _, _ = _core(Xtr[tr], Ctr[tr], ytr[tr], Xtr[va], Ctr[va], int(seed) + 17 * f)
        oof_two[va] = two
        oof_flat[va] = flat
        oof_gsize[va] = gsz
    a = np.clip(oof_flat, 0.0, None)
    b = np.clip(oof_two, 0.0, None)
    d = b - a
    den = float(np.dot(d, d))
    w = float(np.dot(ytr - a, d) / den) if den > 1e-9 else 0.0
    w = float(min(1.0, max(0.0, w)))
    grid = {}
    for wv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        grid[str(wv)] = round(_rmse(wv * b + (1.0 - wv) * a, ytr), 4)
    blend_oof = w * b + (1.0 - w) * a
    idx = {e: i for i, e in enumerate(els)}
    pres = Ctr > TOL
    def col(name):
        j = idx.get(name, -1)
        return pres[:, j] if j >= 0 else np.zeros(n, dtype=bool)
    cup = col("Cu") & col("O")
    feb = col("Fe") & (~cup)
    oth = (~cup) & (~feb)
    chem = {}
    for nm, m in [("cuprate", cup), ("fe_based", feb), ("other", oth)]:
        if int(m.sum()) > 0:
            chem[nm] = {"n": int(m.sum()), "mean_y": round(float(ytr[m].mean()), 3),
                        "flat": round(_rmse(a[m], ytr[m]), 4),
                        "two": round(_rmse(b[m], ytr[m]), 4),
                        "blend": round(_rmse(blend_oof[m], ytr[m]), 4)}
    sz = {}
    for nm, m in [("singleton_group", oof_gsize <= 1.5), ("multi_row_group", oof_gsize > 1.5)]:
        if int(m.sum()) > 0:
            sz[nm] = {"n": int(m.sum()), "flat": round(_rmse(a[m], ytr[m]), 4),
                      "two": round(_rmse(b[m], ytr[m]), 4),
                      "blend": round(_rmse(blend_oof[m], ytr[m]), 4)}
    two_e, flat_e, gsz_e, ng_tr, ng_ev = _core(Xtr, Ctr, ytr, Xev, Cev, int(seed) + 101)
    two_e = np.clip(two_e, 0.0, None)
    flat_e = np.clip(flat_e, 0.0, None)
    pred = np.clip(w * two_e + (1.0 - w) * flat_e, 0.0, None)
    findings = {
        "design": "group key = set of elements with C>0, derived from C on both sides; group model on group-mean features + presence -> group-mean y weighted by row count; deviation model on (row - group mean) + presence -> y - group-mean y; blended with flat HistGB on [X,C]",
        "transductive_note": "evaluation group means/presence are computed from evaluation features only (no labels, no groups, no ids); train-side CV mimics this by computing held-out group means from held-out features only",
        "cv": {"scheme": "GroupKFold on element-set key", "folds": nsp,
                "oof_rmse_flat": round(_rmse(a, ytr), 4),
                "oof_rmse_two_level": round(_rmse(b, ytr), 4),
                "oof_rmse_blend": round(_rmse(blend_oof, ytr), 4),
                "w_two_level_closed_form": round(w, 4),
                "oof_rmse_by_w": grid},
        "oof_by_chemistry": chem,
        "oof_by_group_size": sz,
        "structure": {"train_rows": int(n), "train_groups": int(ng_tr),
                       "eval_rows": int(Xev.shape[0]), "eval_groups": int(ng_ev),
                       "eval_singleton_group_rows": int(np.sum(gsz_e <= 1.5)),
                       "eval_median_rows_per_group_row_weighted": float(np.median(gsz_e))},
        "interpretation": "CALCULATED numbers above are train-only out-of-fold; whether the system-level split helps on development is measured separately. If w is near 0 the two-level arm failed and the measurement is effectively the flat learner."
    }
    return {"prediction": [float(v) for v in pred], "findings": findings}
