import json
import numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

P = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
         l2_regularization=1.0, random_state=1729)


def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def gmae(y, p, g):
    d = defaultdict(list)
    for yi, pi, gi in zip(y, p, g):
        d[gi].append(abs(float(yi) - float(pi)))
    return float(np.mean([np.mean(v) for v in d.values()]))


def r3(x):
    try:
        return float(round(float(x), 3))
    except Exception:
        return None


def analyze(data, history, seed):
    tr = data["train"]; dv = data["development"]
    Xtr = np.asarray(tr["X"], float); Ctr = np.asarray(tr["C"], float)
    ytr = np.asarray(tr["y"], float)
    gtr = np.asarray([str(s) for s in tr["groups"]])
    Xdv = np.asarray(dv["X"], float); Cdv = np.asarray(dv["C"], float)
    ydv = np.asarray(dv["y"], float)
    gdv = np.asarray([str(s) for s in dv["groups"]])

    out = {"calculated": {}, "interpretation": {}}
    c = out["calculated"]
    c["shapes"] = {"train_rows": int(Xtr.shape[0]), "nX": int(Xtr.shape[1]),
                   "nC": int(Ctr.shape[1]), "dev_rows": int(Xdv.shape[0]),
                   "train_groups": int(len(set(gtr.tolist()))),
                   "dev_groups": int(len(set(gdv.tolist())))}
    c["y_pct_0_10_50_90_99_100"] = {
        "train": [r3(v) for v in np.percentile(ytr, [0, 10, 50, 90, 99, 100])],
        "dev": [r3(v) for v in np.percentile(ydv, [0, 10, 50, 90, 99, 100])]}
    c["dev_sd"] = r3(np.std(ydv))

    def floor(C, y):
        keys = defaultdict(list)
        for i in range(C.shape[0]):
            keys[tuple(np.round(C[i], 6).tolist())].append(float(y[i]))
        ss = 0.0; n = 0; dup = 0
        for v in keys.values():
            a = np.asarray(v, float)
            if a.size > 1:
                dup += int(a.size)
            ss += float(np.sum((a - a.mean()) ** 2)); n += int(a.size)
        return {"unique_comps": int(len(keys)), "rows_in_repeated_comps": dup,
                "within_comp_rmse_floor": r3(np.sqrt(ss / max(n, 1)))}

    c["repeat_floor_train"] = floor(Ctr, ytr)
    c["repeat_floor_dev"] = floor(Cdv, ydv)

    ddv = defaultdict(list)
    for i, gi in enumerate(gdv):
        ddv[gi].append(i)
    gmean = {k: float(np.mean(ydv[idx])) for k, idx in ddv.items()}
    oracle = np.array([gmean[g] for g in gdv])
    c["dev_oracle_elementset_mean_rmse"] = r3(rmse(ydv, oracle))

    XCtr = np.hstack([Xtr, Ctr]); XCdv = np.hstack([Xdv, Cdv])
    sets = {"X": (Xtr, Xdv), "C": (Ctr, Cdv), "XC": (XCtr, XCdv)}
    scores = {}; preds = {}
    for name, (A, B) in sets.items():
        m = HistGradientBoostingRegressor(**P).fit(A, ytr)
        p = m.predict(B)
        preds[name] = p
        scores[name] = {"rmse": r3(rmse(ydv, p)), "mae": r3(mae(ydv, p)),
                        "gmae": r3(gmae(ydv, p, gdv))}
    try:
        m = HistGradientBoostingRegressor(**P).fit(XCtr, np.log1p(np.clip(ytr, 0, None)))
        pl = np.clip(np.expm1(m.predict(XCdv)), 0, None)
        scores["XC_log1p"] = {"rmse": r3(rmse(ydv, pl)), "mae": r3(mae(ydv, pl)),
                              "gmae": r3(gmae(ydv, pl, gdv))}
        pb = 0.5 * (preds["XC"] + pl)
        scores["XC_blend_raw_log"] = {"rmse": r3(rmse(ydv, pb)), "mae": r3(mae(ydv, pb)),
                                      "gmae": r3(gmae(ydv, pb, gdv))}
    except Exception as e:
        scores["XC_log1p"] = {"error": str(e)[:120]}
    try:
        m = HistGradientBoostingRegressor(loss="absolute_error", **P).fit(XCtr, ytr)
        pa = m.predict(XCdv)
        scores["XC_absloss"] = {"rmse": r3(rmse(ydv, pa)), "mae": r3(mae(ydv, pa)),
                                "gmae": r3(gmae(ydv, pa, gdv))}
    except Exception as e:
        scores["XC_absloss"] = {"error": str(e)[:120]}
    try:
        sc = StandardScaler().fit(Xtr)
        kn = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(sc.transform(Xtr), ytr)
        pk = kn.predict(sc.transform(Xdv))
        scores["knn5_stdX"] = {"rmse": r3(rmse(ydv, pk)), "mae": r3(mae(ydv, pk)),
                               "gmae": r3(gmae(ydv, pk, gdv))}
    except Exception as e:
        scores["knn5_stdX"] = {"error": str(e)[:120]}
    c["dev_scores"] = scores

    try:
        gkf = GroupKFold(n_splits=5)
        oof = np.zeros(ytr.shape[0])
        for a, b in gkf.split(XCtr, ytr, groups=gtr):
            mm = HistGradientBoostingRegressor(**P).fit(XCtr[a], ytr[a])
            oof[b] = mm.predict(XCtr[b])
        c["train_groupcv_XC"] = {"rmse": r3(rmse(ytr, oof)), "mae": r3(mae(ytr, oof)),
                                 "gmae": r3(gmae(ytr, oof, gtr))}
    except Exception as e:
        c["train_groupcv_XC"] = {"error": str(e)[:120]}

    p = preds["XC"]
    edges = [0.0, 1.0, 5.0, 20.0, 50.0, 90.0, 1e9]
    rb = []
    tot_sq = float(np.sum((p - ydv) ** 2))
    for i in range(len(edges) - 1):
        m_ = (ydv >= edges[i]) & (ydv < edges[i + 1])
        if int(m_.sum()) > 0:
            rb.append({"lo": edges[i], "hi": (edges[i + 1] if edges[i + 1] < 1e8 else -1),
                       "n": int(m_.sum()), "rmse": r3(rmse(ydv[m_], p[m_])),
                       "mean_bias": r3(np.mean(p[m_] - ydv[m_])),
                       "share_sq_err": r3(float(np.sum((p[m_] - ydv[m_]) ** 2)) / max(tot_sq, 1e-9))})
    c["dev_error_by_y_bin_XC"] = rb

    try:
        dtr = defaultdict(list)
        for i, gi in enumerate(gtr):
            dtr[gi].append(i)
        ktr = list(dtr.keys()); kdv = list(ddv.keys())
        Btr = np.array([(Ctr[dtr[k]] > 0).any(axis=0) for k in ktr]).astype(float)
        Bdv = np.array([(Cdv[ddv[k]] > 0).any(axis=0) for k in kdv]).astype(float)
        Mtr = np.array([Ctr[dtr[k]].mean(axis=0) for k in ktr])
        Mdv = np.array([Cdv[ddv[k]].mean(axis=0) for k in kdv])
        inter = Bdv @ Btr.T
        uni = Bdv.sum(1)[:, None] + Btr.sum(1)[None, :] - inter
        jac = inter / np.clip(uni, 1e-9, None)
        maxj = jac.max(1)
        d2 = ((Mdv ** 2).sum(1)[:, None] + (Mtr ** 2).sum(1)[None, :] - 2.0 * (Mdv @ Mtr.T))
        mind = np.sqrt(np.clip(d2.min(1), 0, None))
        gerr = np.array([float(np.mean(np.abs(p[ddv[k]] - ydv[ddv[k]]))) for k in kdv])
        gsq = np.array([float(np.sum((p[ddv[k]] - ydv[ddv[k]]) ** 2)) for k in kdv])
        seen = Btr.any(axis=0) > 0
        unseen_elem_groups = int(np.sum([(Bdv[i] > 0) & (~seen)].__len__() and np.any((Bdv[i] > 0) & (~seen)) for i in range(Bdv.shape[0])))
        ter = np.quantile(maxj, [1.0 / 3.0, 2.0 / 3.0])
        lo = gerr[maxj <= ter[0]]; mid = gerr[(maxj > ter[0]) & (maxj <= ter[1])]; hi = gerr[maxj > ter[1]]
        order = np.argsort(-gsq)[:8]
        c["transfer_distance"] = {
            "spearman_groupMAE_vs_1minusJaccard": r3(spearmanr(1.0 - maxj, gerr).correlation),
            "spearman_groupMAE_vs_meanCompDist": r3(spearmanr(mind, gerr).correlation),
            "maxJaccard_pct_10_50_90": [r3(v) for v in np.percentile(maxj, [10, 50, 90])],
            "groupMAE_by_jaccard_tercile_low_mid_high": [r3(np.mean(lo)) if lo.size else None,
                                                          r3(np.mean(mid)) if mid.size else None,
                                                          r3(np.mean(hi)) if hi.size else None],
            "dev_groups_with_element_absent_from_train": unseen_elem_groups,
            "worst_groups": [{"g": str(kdv[i])[:40], "n": int(len(ddv[kdv[i]])),
                               "mae": r3(gerr[i]), "share_sq": r3(gsq[i] / max(tot_sq, 1e-9)),
                               "maxJ": r3(maxj[i])} for i in order]}
    except Exception as e:
        c["transfer_distance"] = {"error": str(e)[:160]}

    out["interpretation"] = {
        "note": "All numbers above are computed on the supplied train/development rows; statements below are my reading, not measurements.",
        "decision_use": "Compare X vs C vs XC dev RMSE to decide whether to invest in learned element representations (C-side) or descriptor/physics transforms (X-side); compare dev_oracle_elementset_mean_rmse and repeat_floor_dev to total RMSE to bound how much headroom exists; use error-by-y-bin bias and jaccard-tercile group MAE to decide between a two-regime/transformed-target model and a distance-aware robust model; compare train_groupcv_XC to dev XC to judge whether grouped CV on train is a usable internal selection proxy."}

    s = json.dumps(out)
    if len(s.encode("utf-8")) > 8000:
        c.pop("y_pct_0_10_50_90_99_100", None)
        if isinstance(c.get("transfer_distance"), dict):
            c["transfer_distance"].pop("worst_groups", None)
        s = json.dumps(out)
    return json.loads(s)
