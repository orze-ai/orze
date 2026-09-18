import json
import math
import numpy as np

FB = {"used": False, "reason": ""}


def arr(v):
    return np.asarray(v, dtype=float)


def ridge_w(Z, t, alpha):
    d = Z.shape[1]
    A = Z.T.dot(Z) + alpha * np.eye(d)
    b = Z.T.dot(t)
    try:
        return np.linalg.solve(A, b)
    except Exception:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def group_index(groups):
    order = []
    mp = {}
    for i, g in enumerate(groups):
        k = str(g)
        if k not in mp:
            mp[k] = []
            order.append(k)
        mp[k].append(i)
    return order, mp


def rowfeat(Xm, iage, isex, vidx):
    return np.column_stack([Xm[:, iage], Xm[:, isex], Xm[:, vidx]])


def build(X, y, idxlist, vidx, iage, isex, itime):
    F = []
    T = []
    TM = []
    for ii in idxlist:
        r = [float(X[ii[0], iage]), float(X[ii[0], isex])]
        r.extend([float(z) for z in np.median(X[np.ix_(ii, vidx)], axis=0)])
        F.append(r)
        T.append(float(np.mean(y[ii])))
        TM.append(float(np.median(X[ii, itime])))
    return np.asarray(F, dtype=float), np.asarray(T, dtype=float), np.asarray(TM, dtype=float)


def pooled_slope(X, y, idxlist, itime):
    num = 0.0
    den = 0.0
    for ii in idxlist:
        t = X[ii, itime]
        yy = y[ii]
        tc = t - np.mean(t)
        num += float(np.sum(tc * (yy - np.mean(yy))))
        den += float(np.sum(tc * tc))
    if den <= 0.0:
        return 0.0
    return num / den


def mean_abs_dev(M):
    if M.shape[0] < 2:
        return None
    med = np.median(M, axis=0)
    return np.mean(np.abs(M - med), axis=0)


def fit_predict(train, inputs, seed=0):
    names = list(train["feature_names"])
    X = arr(train["X"])
    y = arr(train["y"])
    Xe = arr(inputs["X"])
    iage, isex, itime = 0, 1, 2
    try:
        iage = names.index("age")
        isex = names.index("sex")
        itime = names.index("test_time")
    except Exception:
        iage, isex, itime = 0, 1, 2
        FB["used"] = True
        FB["reason"] = "feature name lookup failed, positional fallback for age/sex/test_time"
    vidx = [j for j in range(len(names)) if j not in (iage, isex, itime)]
    groups = train.get("groups", None)
    if groups is None or len(groups) != X.shape[0]:
        FB["used"] = True
        FB["reason"] = "train groups unavailable, constant row-median fallback used"
        c = float(np.median(y))
        return {"prediction": [c] * int(Xe.shape[0]),
                "findings": {"fallback": FB, "method": "constant row median"}}
    order, mp = group_index(groups)
    idxlist = [np.asarray(mp[k], dtype=int) for k in order]
    P = len(idxlist)
    alphas = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0]
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    tms = [0.0, 1.0]
    paths = ["row", "cluster"]
    sse = {}
    cnt = 0
    for p in range(P):
        tr = [idxlist[q] for q in range(P) if q != p]
        F, T, TM = build(X, y, tr, vidx, iage, isex, itime)
        mu = F.mean(axis=0)
        sd = F.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        base = float(T.mean())
        b = pooled_slope(X, y, tr, itime)
        allrows = np.concatenate(tr)
        tcenter = float(np.mean(X[allrows, itime]))
        medconst = float(np.median(y[allrows]))
        ho = idxlist[p]
        yh = y[ho]
        cnt += int(len(ho))
        Rrow = (rowfeat(X[ho], iage, isex, vidx) - mu) / sd
        cf = np.concatenate([np.asarray([X[ho[0], iage], X[ho[0], isex]], dtype=float),
                             np.median(X[np.ix_(ho, vidx)], axis=0)])
        Rcl = np.tile((cf - mu) / sd, (int(len(ho)), 1))
        tt = X[ho, itime]
        tcl = float(np.median(tt))
        Zc = (F - mu) / sd
        key0 = ("medianconst", 0.0, 0.0, 0.0)
        sse[key0] = sse.get(key0, 0.0) + float(np.sum((medconst - yh) ** 2))
        for a in alphas:
            w = ridge_w(Zc, T - base, a)
            dr = Rrow.dot(w)
            dc = Rcl.dot(w)
            for pa in paths:
                dd = dr if pa == "row" else dc
                ctr = tcenter if pa == "row" else tcl
                for lam in lams:
                    for tm in tms:
                        pred = base + lam * dd + tm * b * (tt - ctr)
                        k = (pa, a, lam, tm)
                        sse[k] = sse.get(k, 0.0) + float(np.sum((pred - yh) ** 2))
    denom = float(max(cnt, 1))
    rows = []
    for k in sse:
        rows.append({"path": k[0], "alpha": float(k[1]), "lam": float(k[2]),
                     "tmult": float(k[3]), "lopo_rmse": math.sqrt(sse[k] / denom)})
    cand = [r for r in rows if r["path"] in ("row", "cluster")]
    cand.sort(key=lambda r: r["lopo_rmse"])
    best = cand[0]
    const_eq = None
    for r in cand:
        if r["lam"] == 0.0 and r["tmult"] == 0.0 and r["path"] == "row":
            const_eq = r["lopo_rmse"]
            break
    const_med = None
    for r in rows:
        if r["path"] == "medianconst":
            const_med = r["lopo_rmse"]
    F, T, TM = build(X, y, idxlist, vidx, iage, isex, itime)
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    base = float(T.mean())
    b = pooled_slope(X, y, idxlist, itime)
    tcenter = float(np.mean(X[:, itime]))
    w = ridge_w((F - mu) / sd, T - base, best["alpha"])
    trmad = []
    for ii in idxlist:
        m = mean_abs_dev(X[np.ix_(ii, vidx)])
        if m is not None:
            trmad.append(m)
    trmad = np.median(np.asarray(trmad, dtype=float), axis=0) if len(trmad) > 0 else None
    clusters = {}
    n_eval = int(Xe.shape[0])
    for i in range(n_eval):
        kk = (round(float(Xe[i, iage]), 3), round(float(Xe[i, isex]), 3))
        if kk not in clusters:
            clusters[kk] = []
        clusters[kk].append(i)
    sizes = sorted([len(clusters[kk]) for kk in clusters])
    ratios = []
    if trmad is not None:
        safe = np.where(trmad < 1e-12, 1e-12, trmad)
        for kk in clusters:
            ii = np.asarray(clusters[kk], dtype=int)
            m = mean_abs_dev(Xe[np.ix_(ii, vidx)])
            if m is not None:
                ratios.append(float(np.median(m / safe)))
    disp_ratio = float(np.median(np.asarray(ratios))) if len(ratios) > 0 else None
    gate_ok = (len(sizes) > 0 and sizes[0] >= 10 and
               len(clusters) <= max(1, n_eval // 10) and
               (disp_ratio is None or disp_ratio <= 1.6))
    use_cluster = (best["path"] == "cluster") and gate_ok
    if best["path"] == "cluster" and not use_cluster:
        FB["used"] = True
        FB["reason"] = "cluster path selected by LOPO but (age,sex) patient-recovery gate failed; row-wise path used instead"
    lam = best["lam"]
    tm = best["tmult"]
    pred = np.zeros(n_eval, dtype=float)
    if use_cluster:
        for kk in clusters:
            ii = np.asarray(clusters[kk], dtype=int)
            cf = np.concatenate([np.asarray([Xe[ii[0], iage], Xe[ii[0], isex]], dtype=float),
                                 np.median(Xe[np.ix_(ii, vidx)], axis=0)])
            dev = float(((cf - mu) / sd).dot(w))
            tcl = float(np.median(Xe[ii, itime]))
            pred[ii] = base + lam * dev + tm * b * (Xe[ii, itime] - tcl)
    else:
        R = (rowfeat(Xe, iage, isex, vidx) - mu) / sd
        pred = base + lam * R.dot(w) + tm * b * (Xe[:, itime] - tcenter)
    bad = np.logical_not(np.isfinite(pred))
    n_bad = int(np.sum(bad))
    if n_bad > 0:
        pred[bad] = base
        FB["used"] = True
        FB["reason"] = (FB["reason"] + " | " if FB["reason"] else "") + \
            "non-finite predictions replaced by equal-patient mean constant"
    findings = {
        "design": "between-patient ridge on patient medians (age,sex,16 voice) + LOPO-chosen shrinkage lam + optional pooled within-patient test_time slope",
        "lopo": {"n_patients": int(P), "n_rows": int(cnt)},
        "selected": {"path": best["path"], "alpha": best["alpha"], "lam": best["lam"],
                     "tmult": best["tmult"], "lopo_rmse": best["lopo_rmse"]},
        "reference_lopo_rmse": {"equal_patient_mean_constant": const_eq,
                                "row_median_constant": const_med},
        "relative_gain_vs_constant": (None if not const_eq else
                                      float((const_eq - best["lopo_rmse"]) / const_eq)),
        "top_combos": cand[:8],
        "pooled_within_patient_time_slope": float(b),
        "applied_path": "cluster_transductive" if use_cluster else "rowwise",
        "transductive_note": "evaluation rows carry no group labels; the cluster path groups unlabeled evaluation rows by exact (age,sex) and uses their medians. This is transductive use of unlabeled inputs only, never labels. Its LOPO estimate assumes perfect patient recovery and is an upper bound.",
        "eval_cluster_diagnostics": {"n_clusters": int(len(clusters)),
                                     "min_size": (sizes[0] if sizes else None),
                                     "max_size": (sizes[-1] if sizes else None),
                                     "n_eval_rows": n_eval,
                                     "within_cluster_dispersion_ratio_vs_train_patients": disp_ratio,
                                     "gate_ok": bool(gate_ok)},
        "caveats": "LOPO also chose the hyperparameters, so its RMSE is mildly optimistic; 7 held-out patients are a small group-level sample; no evaluation-label metric is computed here.",
        "fallback": FB,
        "n_nonfinite_replaced": n_bad
    }
    return {"prediction": [float(v) for v in pred], "findings": json.loads(json.dumps(findings))}
