import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import nnls

SYM = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"]

FAM = {
    "alkali": [3, 11, 19, 37, 55],
    "alkaline_earth": [4, 12, 20, 38, 56],
    "rare_earth": [21, 39] + list(range(57, 72)),
    "lanthanide": list(range(57, 72)),
    "tm3d": list(range(21, 31)),
    "tm4d": list(range(39, 49)),
    "tm5d": [57] + list(range(72, 81)),
    "chalcogen": [8, 16, 34, 52, 84],
    "oxygen": [8],
    "halogen": [9, 17, 35, 53, 85],
    "pnictogen": [7, 15, 33, 51, 83],
    "group13": [5, 13, 31, 49, 81],
    "group14": [6, 14, 32, 50, 82],
    "hydrogen": [1],
    "noble": [2, 10, 18, 36, 54, 86],
    "copper": [29],
    "iron": [26],
    "heavy_p_metal": [81, 82, 83],
    "light_metalloid": [5, 6, 14],
}
FAM_KEYS = sorted(FAM.keys())


def _zs(elements, ncol):
    up = {}
    for i, s in enumerate(SYM):
        up[s.upper()] = i + 1
    zs = []
    ok = 0
    if elements is None:
        elements = []
    for j, e in enumerate(list(elements)):
        key = str(e).strip().upper()
        if key in up:
            zs.append(up[key])
            ok += 1
        else:
            zs.append(j + 1)
    if len(elements) == 0 or ok < 0.5 * max(1, len(elements)):
        zs = [j + 1 for j in range(ncol)]
    if len(zs) < ncol:
        zs = zs + [k + 1 for k in range(len(zs), ncol)]
    return zs[:ncol]


def build(Xl, Cl, elements):
    X = np.asarray(Xl, dtype=float)
    C = np.asarray(Cl, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.clip(C, 0.0, None)
    zs = _zs(elements, C.shape[1])
    M = np.zeros((C.shape[1], len(FAM_KEYS)), dtype=float)
    for k, f in enumerate(FAM_KEYS):
        s = set(FAM[f])
        for j, z in enumerate(zs):
            if z in s:
                M[j, k] = 1.0
    tot = np.clip(C.sum(axis=1, keepdims=True), 1e-12, None)
    P = C / tot
    ff = P.dot(M)
    pres = (C > 1e-9).astype(float)
    fc = pres.dot(M)
    Pc = np.clip(P, 1e-12, None)
    ent = -np.sum(np.where(P > 1e-12, P * np.log(Pc), 0.0), axis=1)
    S = -np.sort(-P, axis=1)
    if S.shape[1] < 3:
        S = np.hstack([S, np.zeros((S.shape[0], 3 - S.shape[1]))])
    nel = pres.sum(axis=1)
    mn = np.where(C > 1e-9, P, np.inf).min(axis=1)
    mn = np.where(np.isfinite(mn), mn, 0.0)
    an = ff[:, FAM_KEYS.index("chalcogen")] + ff[:, FAM_KEYS.index("halogen")]
    ex = np.column_stack([ent, S[:, 0], S[:, 1], S[:, 2], S[:, 0] - S[:, 1], nel, mn, an, an / (1.0 - an + 1e-6), tot.ravel()])
    F = np.hstack([X, C, ff, fc, ex])
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, C, zs


def _classes(C, zs):
    n = C.shape[0]
    pos = {}
    for j, z in enumerate(zs):
        pos[z] = j

    def has(z):
        j = pos.get(z)
        if j is None or j >= C.shape[1]:
            return np.zeros(n, dtype=bool)
        return C[:, j] > 1e-9

    cu = has(29)
    ox = has(8)
    fe = has(26)
    pn = has(33) | has(15) | has(16) | has(34)
    bo = has(5)
    hy = has(1)
    lab = np.array(["other"] * n, dtype=object)
    lab[ox & ~cu] = "oxide_other"
    lab[hy & ~cu] = "hydride_like"
    lab[bo & ~cu & ~ox] = "boride"
    lab[fe & pn] = "fe_based"
    lab[cu & ox] = "cuprate"
    return lab


def _hgb(s):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=s)


def fit_a(F, y, s):
    return _hgb(s).fit(F, y)


def pred_a(m, F):
    return m.predict(F)


def fit_gate(F, y, s):
    t = 30.0
    lab = (y >= t).astype(int)
    if lab.sum() < 50 or (len(lab) - lab.sum()) < 50:
        return (None, _hgb(s).fit(F, y), None)
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=s)
    clf.fit(F, lab)
    lo = _hgb(s).fit(F[lab == 0], y[lab == 0])
    hi = _hgb(s).fit(F[lab == 1], y[lab == 1])
    return (clf, lo, hi)


def pred_gate(m, F):
    clf, lo, hi = m
    if clf is None:
        return lo.predict(F)
    p = clf.predict_proba(F)[:, 1]
    return (1.0 - p) * lo.predict(F) + p * hi.predict(F)


def fit_sqrt(F, y, s):
    ys = np.sqrt(np.clip(y, 0.0, None))
    m = _hgb(s).fit(F, ys)
    r = ys - m.predict(F)
    return (m, float(np.var(r)))


def pred_sqrt(mv, F):
    m, v = mv
    q = np.clip(m.predict(F), 0.0, None)
    return q * q + v


def fit_et(F, y, s):
    return ExtraTreesRegressor(n_estimators=150, max_features=0.4, min_samples_leaf=2, random_state=s, n_jobs=1).fit(F, y)


def pred_et(m, F):
    return m.predict(F)


def _rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def _gmae(a, b, g):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = {}
    for i in range(len(a)):
        d.setdefault(g[i], []).append(abs(a[i] - b[i]))
    return float(np.mean([float(np.mean(v)) for v in d.values()]))


def fit_predict(train, inputs, seed=0):
    try:
        rs = int(seed)
    except Exception:
        rs = 0
    y = np.asarray(train["y"], dtype=float)
    el_tr = train.get("elements")
    el_te = inputs.get("elements") or el_tr
    Ftr, Ctr, zs = build(train["X"], train["C"], el_tr)
    Fte, Cte, _ = build(inputs["X"], inputs["C"], el_te)
    if Fte.shape[1] != Ftr.shape[1]:
        k = min(Fte.shape[1], Ftr.shape[1])
        Ftr = Ftr[:, :k]
        Fte = Fte[:, :k]
    groups = train.get("groups")
    if groups is None:
        groups = ["g%d" % i for i in range(len(y))]
    groups = [str(g) for g in groups]
    ymax = float(np.max(y)) if len(y) else 100.0

    specs = [("A_hgb_raw", fit_a, pred_a), ("B_gate30", fit_gate, pred_gate), ("C_sqrt", fit_sqrt, pred_sqrt), ("D_extratrees", fit_et, pred_et)]
    ng = len(set(groups))
    nsp = 4 if ng >= 8 else 2
    try:
        folds = list(GroupKFold(n_splits=nsp).split(Ftr, y, groups=np.asarray(groups)))
    except Exception:
        folds = []
    oof = {}
    errs = {}
    for name, ff, pf in specs:
        if not folds:
            break
        try:
            col = np.zeros(len(y), dtype=float)
            for tr, va in folds:
                mm = ff(Ftr[tr], y[tr], rs)
                col[va] = pf(mm, Ftr[va])
            oof[name] = np.clip(col, 0.0, 1.2 * ymax)
        except Exception as e:
            errs[name] = str(e)[:160]
    fitters = {"A_hgb_raw": (fit_a, pred_a), "B_gate30": (fit_gate, pred_gate), "C_sqrt": (fit_sqrt, pred_sqrt), "D_extratrees": (fit_et, pred_et)}
    if not oof:
        mm = fit_a(Ftr, y, rs)
        pr = np.clip(pred_a(mm, Fte), 0.0, 1.05 * ymax)
        return {"prediction": [float(v) for v in pr], "findings": {"status": "cv_unavailable_fallback_single_hgb", "errors": errs}}

    names = [n for n, _, _ in specs if n in oof]
    Mo = np.column_stack([oof[n] for n in names])
    try:
        w, _ = nnls(Mo, y)
    except Exception:
        w = np.zeros(len(names))
    if float(np.sum(w)) <= 1e-9:
        w = np.ones(len(names)) / float(len(names))
    blend_oof = np.clip(Mo.dot(w), 0.0, 1.2 * ymax)

    metrics = {}
    for n in names:
        metrics[n] = {"cv_rmse": round(_rmse(y, oof[n]), 4), "cv_mae": round(_mae(y, oof[n]), 4), "cv_group_mae": round(_gmae(y, oof[n], groups), 4)}
    metrics["E_nnls_blend"] = {"cv_rmse": round(_rmse(y, blend_oof), 4), "cv_mae": round(_mae(y, blend_oof), 4), "cv_group_mae": round(_gmae(y, blend_oof, groups), 4)}

    order = sorted(metrics.keys(), key=lambda k: metrics[k]["cv_rmse"])
    chosen = order[0]

    edges = [0.0, 10.0, 30.0, 60.0, 100.0, 1e9]
    bins = {}
    for bi in range(len(edges) - 1):
        msk = (y >= edges[bi]) & (y < edges[bi + 1])
        if int(msk.sum()) == 0:
            continue
        key = "y_%g_%g" % (edges[bi], edges[bi + 1] if edges[bi + 1] < 1e8 else 999)
        ent = {"n": int(msk.sum())}
        for n in names:
            ent[n] = round(_rmse(y[msk], oof[n][msk]), 3)
        ent["E_nnls_blend"] = round(_rmse(y[msk], blend_oof[msk]), 3)
        bins[key] = ent

    lab = _classes(Ctr, zs)
    cls = {}
    for c in sorted(set([str(v) for v in lab])):
        msk = np.array([str(v) == c for v in lab])
        if int(msk.sum()) < 20:
            continue
        ent = {"n": int(msk.sum()), "mean_y": round(float(np.mean(y[msk])), 2)}
        ent["A_hgb_raw"] = round(_rmse(y[msk], oof["A_hgb_raw"][msk]), 3) if "A_hgb_raw" in oof else None
        ent[chosen] = round(_rmse(y[msk], (blend_oof if chosen == "E_nnls_blend" else oof[chosen])[msk]), 3)
        cls[c] = ent

    if chosen == "E_nnls_blend":
        cols = []
        for n in names:
            ff, pf = fitters[n]
            cols.append(pf(ff(Ftr, y, rs), Fte))
        pred = np.column_stack(cols).dot(w)
    else:
        ff, pf = fitters[chosen]
        pred = pf(ff(Ftr, y, rs), Fte)
    pred = np.clip(np.nan_to_num(np.asarray(pred, dtype=float), nan=float(np.median(y))), 0.0, 1.05 * ymax)

    findings = {
        "design": "representation held fixed (X + C + family aggregates, identical to parent block V4); only learner/loss varies; train-only GroupKFold over element-set groups",
        "acknowledged_transformations": "evaluation features transformed only by deterministic per-row functions of X and C (family fraction/count sums, composition entropy, sorted-fraction order statistics); no labels or cross-row fitting on evaluation data",
        "n_folds": len(folds),
        "n_train_rows": int(len(y)),
        "n_train_groups": int(ng),
        "n_features": int(Ftr.shape[1]),
        "cv_metrics": metrics,
        "cv_rank_by_rmse": order,
        "blend_weights": {names[i]: round(float(w[i]), 4) for i in range(len(names))},
        "cv_rmse_by_target_bin": bins,
        "cv_rmse_by_chemistry_class": cls,
        "chosen_for_submission": chosen,
        "candidate_errors": errs,
        "prediction_stats": {
            "mean": round(float(np.mean(pred)), 3),
            "median": round(float(np.median(pred)), 3),
            "p90": round(float(np.percentile(pred, 90)), 3),
            "max": round(float(np.max(pred)), 3),
            "frac_below_5K": round(float(np.mean(pred < 5.0)), 4),
        },
        "train_target_stats": {"mean": round(float(np.mean(y)), 3), "median": round(float(np.median(y)), 3), "max": round(ymax, 2)},
        "interpretation_unverified": "All numbers above are train-internal grouped CV; they do not establish development or confirmation behaviour. Candidate selection by CV RMSE is a single joint choice, so a development gain cannot be attributed to one mechanism without the reported per-candidate CV and bin/class decomposition being reproduced on labelled data.",
    }
    return {"prediction": [float(v) for v in pred], "findings": findings}
