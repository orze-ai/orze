import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HGB = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
           l2_regularization=1.0, random_state=1729)


def _build(X, C):
    X = np.asarray(X, float)
    C = np.asarray(C, float)
    s = C.sum(1, keepdims=True)
    s[s <= 0] = 1.0
    F = C / s
    P = np.where(F > 0, F, 1.0)
    ent = -(F * np.log(P)).sum(1)
    nel = (F > 1e-9).sum(1).astype(float)
    srt = -np.sort(-F, axis=1)[:, :5]
    ex = np.column_stack([ent, nel, srt, srt[:, 0] - srt[:, 1]])
    Z = np.hstack([X, F, ex])
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), F


def _groups(F):
    keys = [",".join(map(str, np.nonzero(F[i] > 1e-9)[0].tolist()))
            for i in range(F.shape[0])]
    u, inv = np.unique(np.asarray(keys), return_inverse=True)
    return inv.astype(int), u


def _gm(Z, g):
    k = int(g.max()) + 1
    cnt = np.bincount(g, minlength=k).astype(float)
    cnt[cnt <= 0] = 1.0
    S = np.zeros((k, Z.shape[1]))
    np.add.at(S, g, Z)
    Mg = S / cnt[:, None]
    return Mg, cnt, Mg[g]


def _fit_decomp(Z, y, g):
    Mg, cnt, M = _gm(Z, g)
    yg = np.bincount(g, weights=y, minlength=len(cnt)) / cnt
    mb = HistGradientBoostingRegressor(**HGB).fit(Mg, yg, sample_weight=cnt)
    mw = HistGradientBoostingRegressor(**HGB).fit(np.hstack([Z - M, M]), y - yg[g])
    return mb, mw


def _pred_decomp(mb, mw, Z, g):
    if g is None:
        M = Z
    else:
        _, _, M = _gm(Z, g)
    p = mb.predict(M) + mw.predict(np.hstack([Z - M, M]))
    return np.clip(p, 0.0, 200.0)


def _classes(F, elements):
    el = list(elements)
    idx = {}
    for i, e in enumerate(el):
        idx[str(e)] = i
    n = F.shape[0]

    def has(sym):
        j = idx.get(sym)
        if j is None or j >= F.shape[1]:
            return np.zeros(n, bool)
        return F[:, j] > 1e-9

    cu = has("Cu")
    o = has("O")
    fe = has("Fe")
    b = has("B")
    h = has("H")
    pn = has("As") | has("Se") | has("P") | has("Te")
    lab = np.array(["other"] * n, dtype=object)
    lab[b] = "boride"
    lab[h & ~o] = "hydride_like"
    lab[o] = "oxide_other"
    lab[fe & pn] = "fe_based"
    lab[cu & o] = "cuprate"
    return lab


def _met(y, p, g):
    r = np.asarray(p, float) - np.asarray(y, float)
    rmse = float(np.sqrt(np.mean(r ** 2)))
    mae = float(np.mean(np.abs(r)))
    gg = []
    for u in np.unique(g):
        m = g == u
        gg.append(float(np.mean(np.abs(r[m]))))
    return round(rmse, 4), round(mae, 4), round(float(np.mean(gg)), 4)


def fit_predict(train, inputs, seed=0):
    ytr = np.asarray(train["y"], float)
    Ztr, Ftr = _build(train["X"], train["C"])
    gtr, _u = _groups(Ftr)
    Zev, Fev = _build(inputs["X"], inputs["C"])
    gev, _ue = _groups(Fev)
    n = Ztr.shape[0]

    # sanity check: derived element-set groups vs supplied train groups
    agree = None
    try:
        prov = np.asarray([str(s) for s in train["groups"]])
        ok = 0
        for u in np.unique(gtr):
            if len(np.unique(prov[gtr == u])) == 1:
                ok += 1
        agree = {"derived_groups": int(len(np.unique(gtr))),
                 "supplied_groups": int(len(np.unique(prov))),
                 "derived_groups_pure_in_supplied": int(ok)}
    except Exception as e:
        agree = {"error": str(e)[:120]}

    # variance decomposition of train Tc between / within element sets
    Mg_y, cnt, _mrow = _gm(ytr.reshape(-1, 1), gtr)
    yg = Mg_y[:, 0]
    tot = float(np.var(ytr))
    betw = float(np.sum(cnt * (yg - ytr.mean()) ** 2) / n)
    vdec = {"total_var": round(tot, 3),
            "between_set_share": round(betw / tot, 4) if tot > 0 else None,
            "within_set_rms": round(float(np.sqrt(np.mean((ytr - yg[gtr]) ** 2))), 3)}

    names = ["A_flat", "B_trans", "C_norow"]
    oof = {k: np.full(n, np.nan) for k in names}
    gkf = GroupKFold(n_splits=4)
    for tr, te in gkf.split(Ztr, ytr, groups=gtr):
        ma = HistGradientBoostingRegressor(**HGB).fit(Ztr[tr], ytr[tr])
        oof["A_flat"][te] = np.clip(ma.predict(Ztr[te]), 0.0, 200.0)
        _uu, gl = np.unique(gtr[tr], return_inverse=True)
        mb, mw = _fit_decomp(Ztr[tr], ytr[tr], gl.astype(int))
        _uv, gt = np.unique(gtr[te], return_inverse=True)
        oof["B_trans"][te] = _pred_decomp(mb, mw, Ztr[te], gt.astype(int))
        oof["C_norow"][te] = _pred_decomp(mb, mw, Ztr[te], None)

    cv = {}
    for k in names:
        r, m, gmae = _met(ytr, oof[k], gtr)
        cv[k] = {"cv_rmse": r, "cv_mae": m, "cv_group_mae": gmae}

    cls = _classes(Ftr, train["elements"])
    by_cls = {}
    for c in np.unique(cls):
        m = cls == c
        e = {"n": int(m.sum()), "mean_y": round(float(ytr[m].mean()), 2)}
        for k in names:
            e[k] = round(float(np.sqrt(np.mean((oof[k][m] - ytr[m]) ** 2))), 3)
        by_cls[str(c)] = e

    by_gsz = {}
    sz = cnt[gtr]
    for lo, hi, tag in [(1, 1, "size_1"), (2, 4, "size_2_4"),
                        (5, 12, "size_5_12"), (13, 10 ** 6, "size_13p")]:
        m = (sz >= lo) & (sz <= hi)
        if m.sum() > 0:
            e = {"n": int(m.sum())}
            for k in names:
                e[k] = round(float(np.sqrt(np.mean((oof[k][m] - ytr[m]) ** 2))), 3)
            by_gsz[tag] = e

    # final fit on all train, transductive group means on the evaluation pool
    mb, mw = _fit_decomp(Ztr, ytr, gtr)
    pred = _pred_decomp(mb, mw, Zev, gev)
    ma = HistGradientBoostingRegressor(**HGB).fit(Ztr, ytr)
    flat = np.clip(ma.predict(Zev), 0.0, 200.0)

    ev_cnt = np.bincount(gev).astype(float)
    findings = {
        "design": "representation and learner held fixed (81 X + 86 renormalised C + entropy/count/top-5 fraction stats; HistGB 250/0.08/31/l2=1/seed1729 everywhere). Only the target/prediction structure varies: A_flat = flat row model; B_trans = between-set model on group-mean features (weighted by group row count) + within-set deviation model, group means on the evaluation side built from unlabeled features; C_norow = same fitted models with pooling ablated (group mean replaced by the row itself).",
        "submitted": "B_trans",
        "acknowledged_transformations": "evaluation features are transformed by deterministic row functions of X and C and by averaging over evaluation rows sharing the same nonzero-element support derived from C; no evaluation labels, group labels, row ids or formulas are used, and no target information crosses from evaluation rows.",
        "derived_group_check": agree,
        "train_variance_decomposition": vdec,
        "cv_metrics": cv,
        "cv_rmse_by_class": by_cls,
        "cv_rmse_by_train_group_size": by_gsz,
        "group_size_stats": {
            "train_groups": int(len(cnt)),
            "train_mean_rows_per_group": round(float(cnt.mean()), 2),
            "train_frac_singleton": round(float((cnt == 1).mean()), 4),
            "eval_groups": int(len(ev_cnt)),
            "eval_mean_rows_per_group": round(float(ev_cnt.mean()), 2),
            "eval_frac_singleton": round(float((ev_cnt == 1).mean()), 4)},
        "prediction_stats": {
            "mean": round(float(pred.mean()), 3),
            "median": round(float(np.median(pred)), 3),
            "p90": round(float(np.percentile(pred, 90)), 3),
            "max": round(float(pred.max()), 3),
            "frac_below_5K": round(float((pred < 5).mean()), 4),
            "mean_abs_diff_vs_flat": round(float(np.mean(np.abs(pred - flat))), 3),
            "corr_with_flat": round(float(np.corrcoef(pred, flat)[0, 1]), 4)},
        "train_target_stats": {"mean": round(float(ytr.mean()), 3),
                               "median": round(float(np.median(ytr)), 3),
                               "max": round(float(ytr.max()), 3)},
        "interpretation_unverified": "All tables are train-only GroupKFold over element-set groups; they are not development or confirmation measurements. B_trans vs A_flat isolates the decomposition+pooling mechanism at fixed features/learner; B_trans vs C_norow isolates transductive pooling alone with identical fitted models. A development change cannot be attributed to pooling unless the B_trans/C_norow gap reproduces there."}
    s = json.dumps(findings)
    if len(s.encode("utf-8")) > 8000:
        findings.pop("cv_rmse_by_train_group_size", None)
    return {"prediction": [float(v) for v in pred], "findings": findings}
