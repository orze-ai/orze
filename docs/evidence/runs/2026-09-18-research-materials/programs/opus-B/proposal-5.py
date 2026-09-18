import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import nnls

N_FOLDS = 4
SPECS = [("hgb_raw", "raw"), ("hgb_log", "log"), ("et_raw", "raw")]


def _feat(X, C):
    return np.hstack([np.asarray(X, dtype=float), np.asarray(C, dtype=float)])


def _rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _make(name, seed):
    if name == "hgb_raw":
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.07,
                                             max_leaf_nodes=31, min_samples_leaf=15,
                                             l2_regularization=1.0, early_stopping=False,
                                             random_state=int(seed) + 11)
    if name == "hgb_log":
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.07,
                                             max_leaf_nodes=31, min_samples_leaf=15,
                                             l2_regularization=1.0, early_stopping=False,
                                             random_state=int(seed) + 23)
    return ExtraTreesRegressor(n_estimators=150, max_features=0.35, min_samples_leaf=2,
                               bootstrap=False, n_jobs=1, random_state=int(seed) + 37)


def _fit_one(name, kind, Xtr, ytr, Xte, seed):
    est = _make(name, seed)
    if kind == "log":
        est.fit(Xtr, np.log1p(np.clip(ytr, 0.0, None)))
        p = np.expm1(est.predict(Xte))
    else:
        est.fit(Xtr, ytr)
        p = est.predict(Xte)
    p = np.asarray(p, dtype=float)
    p[~np.isfinite(p)] = float(np.mean(ytr))
    return np.clip(p, 0.0, None)


def _nnls_w(P, y):
    try:
        w, _ = nnls(np.asarray(P, dtype=float), np.asarray(y, dtype=float))
    except Exception:
        w = np.ones(P.shape[1]) / float(P.shape[1])
    w = np.asarray(w, dtype=float)
    if (not np.isfinite(w).all()) or float(w.sum()) <= 1e-9:
        w = np.ones(P.shape[1]) / float(P.shape[1])
    return w


def _groups_from_C(C, elements):
    C = np.asarray(C, dtype=float)
    out = []
    for r in C:
        idx = np.nonzero(r > 0.0)[0]
        out.append("-".join([str(elements[i]) for i in idx]))
    return out


def _family_masks(C, elements):
    C = np.asarray(C, dtype=float)
    names = [str(e) for e in elements]
    def col(sym):
        return names.index(sym) if sym in names else -1
    icu, io, ife = col("Cu"), col("O"), col("Fe")
    has = lambda i: (C[:, i] > 0.0) if i >= 0 else np.zeros(C.shape[0], dtype=bool)
    cup = has(icu) & has(io)
    fe = has(ife) & (~cup)
    other = ~(cup | fe)
    return {"cuprate": cup, "fe_based": fe, "other": other}


def fit_predict(train, inputs, seed=0):
    Xtr = _feat(train["X"], train["C"])
    ytr = np.asarray(train["y"], dtype=float)
    Xev = _feat(inputs["X"], inputs["C"])
    elements = list(train["elements"])
    g = train.get("groups", None)
    if g is None:
        g = _groups_from_C(train["C"], elements)
    groups = np.asarray([str(v) for v in g])

    n = Xtr.shape[0]
    m = len(SPECS)
    oof = np.zeros((n, m), dtype=float)
    fold_id = np.full(n, -1, dtype=int)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for k, (tr, te) in enumerate(gkf.split(Xtr, ytr, groups)):
        fold_id[te] = k
        for j in range(m):
            name, kind = SPECS[j]
            oof[te, j] = _fit_one(name, kind, Xtr[tr], ytr[tr], Xtr[te], seed + 100 * k)

    base_rmse = {SPECS[j][0]: round(_rmse(oof[:, j], ytr), 4) for j in range(m)}
    w_full = _nnls_w(oof, ytr)
    blend_oof = oof.dot(w_full)

    hb = np.zeros(n, dtype=float)
    hi = np.zeros(n, dtype=float)
    for k in range(N_FOLDS):
        te = fold_id == k
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        wk = _nnls_w(oof[tr], ytr[tr])
        bt = oof[tr].dot(wk)
        bv = oof[te].dot(wk)
        hb[te] = bv
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(bt, ytr[tr])
        hi[te] = np.clip(iso.predict(bv), 0.0, None)

    rmse_blend_nested = _rmse(hb, ytr)
    rmse_iso_nested = _rmse(hi, ytr)
    use_iso = bool(rmse_iso_nested < rmse_blend_nested - 0.02)

    dec = {}
    order = np.argsort(hb)
    chunks = np.array_split(order, 10)
    for q, ix in enumerate(chunks):
        if len(ix) == 0:
            continue
        dec[str(q)] = {"n": int(len(ix)),
                       "mean_pred": round(float(np.mean(hb[ix])), 3),
                       "mean_y": round(float(np.mean(ytr[ix])), 3),
                       "rmse": round(_rmse(hb[ix], ytr[ix]), 3),
                       "rmse_iso": round(_rmse(hi[ix], ytr[ix]), 3)}

    fam = _family_masks(train["C"], elements)
    fam_out = {}
    for key, msk in fam.items():
        if int(msk.sum()) == 0:
            continue
        best_single = min([_rmse(oof[msk, j], ytr[msk]) for j in range(m)])
        fam_out[key] = {"n": int(msk.sum()),
                        "mean_y": round(float(np.mean(ytr[msk])), 3),
                        "rmse_best_single": round(float(best_single), 3),
                        "rmse_blend": round(_rmse(hb[msk], ytr[msk]), 3),
                        "rmse_blend_iso": round(_rmse(hi[msk], ytr[msk]), 3)}

    final_models = []
    for j in range(m):
        name, kind = SPECS[j]
        final_models.append(_fit_one(name, kind, Xtr, ytr, Xev, seed + 7))
    Pev = np.vstack(final_models).T
    pred = Pev.dot(w_full)
    if use_iso:
        iso_f = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso_f.fit(blend_oof, ytr)
        pred = iso_f.predict(pred)
    pred = np.asarray(pred, dtype=float)
    pred[~np.isfinite(pred)] = float(np.median(ytr))
    pred = np.clip(pred, 0.0, float(np.max(ytr)) * 1.05)

    findings = {
        "design": "flat [X|C] rows; base learners = HistGB(raw y), HistGB(log1p y), ExtraTrees; NNLS blend weights fitted on GroupKFold(element-set) OOF; isotonic recalibration of the blend applied only if a nested fold-wise estimate improves train OOF RMSE by >0.02 K",
        "cv": {"folds": N_FOLDS,
               "group_key": "train element-set groups",
               "oof_rmse_base": base_rmse,
               "nnls_weights": {SPECS[j][0]: round(float(w_full[j]), 4) for j in range(m)},
               "oof_rmse_blend_insample_weights": round(_rmse(blend_oof, ytr), 4),
               "oof_rmse_blend_nested": round(rmse_blend_nested, 4),
               "oof_rmse_blend_iso_nested": round(rmse_iso_nested, 4),
               "isotonic_applied": use_iso},
        "shrinkage_check_deciles_of_nested_blend": dec,
        "by_chemistry": fam_out,
        "transductive_note": "no evaluation labels, groups or ids used; evaluation features enter only through the frozen fitted models, no statistics are pooled across evaluation rows",
        "interpretation": "CALCULATED values are train-only out-of-fold. If nnls weight on hgb_log is ~0 and isotonic is rejected while top-decile mean_pred is close to mean_y, the tail-shrinkage explanation of the cuprate error mass is not supported and the development measurement should land near the existing flat-model level; the development RMSE reported separately is the only out-of-sample evidence."
    }
    return {"prediction": [float(v) for v in pred], "findings": findings}
