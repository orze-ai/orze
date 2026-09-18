import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HGB = {"max_iter": 250, "learning_rate": 0.08, "max_leaf_nodes": 31,
       "l2_regularization": 1.0, "random_state": 1729}


def rmse(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(math.sqrt(float(np.mean((p - y) ** 2))))


def gidx(gs):
    d = {}
    for i, g in enumerate(gs):
        k = str(g)
        if k in d:
            d[k].append(i)
        else:
            d[k] = [i]
    return d


def dup_floor(C, y):
    keys = {}
    for i in range(C.shape[0]):
        k = tuple(np.round(C[i], 4).tolist())
        if k in keys:
            keys[k].append(float(y[i]))
        else:
            keys[k] = [float(y[i])]
    ss = 0.0
    ndf = 0
    nrows_dup = 0
    spread = 0.0
    for k in keys:
        v = np.asarray(keys[k], dtype=float)
        if v.size > 1:
            nrows_dup += int(v.size)
            ss += float(np.sum((v - v.mean()) ** 2))
            ndf += int(v.size) - 1
            sp = float(v.max() - v.min())
            if sp > spread:
                spread = sp
    n = int(C.shape[0])
    return {"unique_compositions": len(keys), "rows_in_dup_groups": nrows_dup,
            "within_dup_rmse": round(math.sqrt(ss / max(ndf, 1)), 3),
            "noise_floor_rmse_all_rows": round(math.sqrt(ss / max(n, 1)), 3),
            "max_spread_K": round(spread, 2)}


def analyze(data, history, seed):
    tr = data["train"]
    dv = data["development"]
    Xtr = np.asarray(tr["X"], dtype=float)
    Ctr = np.asarray(tr["C"], dtype=float)
    ytr = np.asarray(tr["y"], dtype=float)
    Xdv = np.asarray(dv["X"], dtype=float)
    Cdv = np.asarray(dv["C"], dtype=float)
    ydv = np.asarray(dv["y"], dtype=float)
    gtr = [str(g) for g in tr["groups"]]
    gdv = [str(g) for g in dv["groups"]]
    out = {}
    out["shapes"] = {"train_rows": int(Xtr.shape[0]), "dev_rows": int(Xdv.shape[0]),
                     "nX": int(Xtr.shape[1]), "nC": int(Ctr.shape[1]),
                     "train_sets": len(set(gtr)), "dev_sets": len(set(gdv))}
    out["y_stats"] = {"train_mean": round(float(ytr.mean()), 2),
                      "dev_mean": round(float(ydv.mean()), 2),
                      "train_sd": round(float(ytr.std()), 2),
                      "dev_sd": round(float(ydv.std()), 2),
                      "dev_max": round(float(ydv.max()), 2),
                      "train_frac_gt60": round(float(np.mean(ytr > 60.0)), 4),
                      "dev_frac_gt60": round(float(np.mean(ydv > 60.0)), 4)}
    out["dup_train"] = dup_floor(Ctr, ytr)
    out["dup_dev"] = dup_floor(Cdv, ydv)

    dvg = gidx(gdv)
    gy = np.zeros(ydv.shape[0], dtype=float)
    for k in dvg:
        ix = dvg[k]
        gy[ix] = float(np.mean(ydv[ix]))
    out["oracle_setmean"] = {
        "rmse_if_set_mean_known": round(rmse(gy, ydv), 3),
        "between_set_sd_of_setmeans": round(float(np.std(gy)), 3)}

    preds = {}

    def run(name, Ftr, Fdv, logt):
        m = HistGradientBoostingRegressor(**HGB)
        t = np.log1p(np.maximum(ytr, 0.0)) if logt else ytr
        m.fit(Ftr, t)
        p = np.asarray(m.predict(Fdv), dtype=float)
        if logt:
            p = np.expm1(p)
        p = np.where(np.isfinite(p), p, float(np.median(ytr)))
        preds[name] = p
        r = p - ydv
        gm = []
        for kk in dvg:
            gm.append(float(np.mean(np.abs(r[dvg[kk]]))))
        return {"rmse": round(rmse(p, ydv), 3),
                "mae": round(float(np.mean(np.abs(r))), 3),
                "group_mae": round(float(np.mean(np.asarray(gm))), 3),
                "mean_signed_resid": round(float(np.mean(r)), 3)}

    res = {}
    res["X_only"] = run("X", Xtr, Xdv, False)
    res["C_only"] = run("C", Ctr, Cdv, False)
    res["X_plus_C"] = run("XC", np.hstack([Xtr, Ctr]), np.hstack([Xdv, Cdv]), False)
    res["X_only_log1p_target"] = run("Xlog", Xtr, Xdv, True)
    out["models"] = res

    dec = {}
    for nm in ["X", "C", "XC", "Xlog"]:
        p = preds[nm]
        gp = np.zeros(p.shape[0], dtype=float)
        for k in dvg:
            ix = dvg[k]
            gp[ix] = float(np.mean(p[ix]))
        btw = rmse(gp, gy)
        wth = rmse((p - gp), (ydv - gy))
        dec[nm] = {"total_rmse": round(rmse(p, ydv), 3),
                   "between_set_rmse": round(btw, 3),
                   "within_set_rmse": round(wth, 3),
                   "between_share_of_mse": round(btw * btw / max(btw * btw + wth * wth, 1e-9), 4)}
    out["decomposition"] = dec

    edges = [0.0, 10.0, 20.0, 40.0, 60.0, 90.0, 1e9]
    tail = {}
    pX = preds["X"]
    pL = preds["Xlog"]
    for j in range(len(edges) - 1):
        sel = (ydv >= edges[j]) & (ydv < edges[j + 1])
        n = int(np.sum(sel))
        lab = str(int(edges[j])) + "_" + ("inf" if j == len(edges) - 2 else str(int(edges[j + 1])))
        if n == 0:
            tail[lab] = {"n": 0}
            continue
        tail[lab] = {"n": n,
                     "X_rmse": round(rmse(pX[sel], ydv[sel]), 2),
                     "X_mean_signed": round(float(np.mean(pX[sel] - ydv[sel])), 2),
                     "Xlog_rmse": round(rmse(pL[sel], ydv[sel]), 2),
                     "Xlog_mean_signed": round(float(np.mean(pL[sel] - ydv[sel])), 2)}
    out["tc_bins"] = tail

    elems = [str(e) for e in tr["elements"]]
    tr_present = (Ctr > 0.0).any(axis=0)
    unseen_rows = int(np.sum(((Cdv > 0.0) & (~tr_present)).any(axis=1)))
    out["unseen_elements"] = {"n_elements_in_train": int(np.sum(tr_present)),
                             "n_elements_total": len(elems),
                             "dev_rows_with_unseen_element": unseen_rows}

    tr_sets = []
    seen = {}
    for g in gtr:
        if g not in seen:
            seen[g] = 1
            tr_sets.append(frozenset(g.split("-")))
    best = {}
    for k in dvg:
        s = frozenset(k.split("-"))
        bj = 0.0
        for t in tr_sets:
            inter = len(s & t)
            if inter == 0:
                continue
            j = inter / float(len(s | t))
            if j > bj:
                bj = j
        best[k] = bj
    jv = np.zeros(ydv.shape[0], dtype=float)
    for k in dvg:
        jv[dvg[k]] = best[k]
    jb = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
    jout = {}
    for j in range(len(jb) - 1):
        sel = (jv >= jb[j]) & (jv < jb[j + 1])
        n = int(np.sum(sel))
        lab = "jac_" + str(jb[j]) + "_" + str(jb[j + 1])
        if n == 0:
            jout[lab] = {"n": 0}
            continue
        jout[lab] = {"n": n,
                     "n_sets": int(len(set([gdv[i] for i in np.nonzero(sel)[0].tolist()]))),
                     "mean_y": round(float(np.mean(ydv[sel])), 2),
                     "X_rmse": round(rmse(pX[sel], ydv[sel]), 2),
                     "X_mae": round(float(np.mean(np.abs(pX[sel] - ydv[sel]))), 2)}
    out["jaccard_bins"] = jout
    out["jaccard_summary"] = {"mean_max_jaccard_over_dev_sets": round(float(np.mean(np.asarray(list(best.values())))), 4),
                              "dev_sets_with_zero_overlap": int(np.sum(np.asarray(list(best.values())) == 0.0))}

    out["interpretation_note"] = "All numbers above are computed from actual train/development rows with the reference HGB hyperparameters; they are diagnostics, not a claim that any representation transfers causally. Learned feature transforms of evaluation inputs were not used here."
    out["history_seen"] = {"n_actions": (len(history["actions"]) if isinstance(history, dict) and "actions" in history else -1)}
    return out
