import math
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

JIT = ["Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP"]
SHI = ["Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA"]
NOI = ["NHR", "HNR"]
NON = ["RPDE", "DFA", "PPE"]
VOICE = JIT + SHI + NOI + NON
LOGF = set(JIT + SHI + ["NHR"])
ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
SHRINKS = [0.0, 0.15, 0.3, 0.5]
SPECS = ["const", "age", "agesex", "core3", "core5", "comp", "compvar", "all16"]
NFEAT = {"const": 0, "age": 1, "agesex": 2, "core3": 3, "core5": 5, "comp": 6, "compvar": 7, "all16": 18}


def _logx(X, names):
    A = np.asarray(X, dtype=float).copy()
    for f in LOGF:
        if f in names:
            j = names.index(f)
            A[:, j] = np.log(np.clip(A[:, j], 1e-10, None))
    return A


def _praw(A, rows, names):
    sub = A[rows, :]
    d = {"age": float(np.median(sub[:, names.index("age")])),
         "sex": float(np.median(sub[:, names.index("sex")]))}
    for f in VOICE:
        v = sub[:, names.index(f)]
        d["med_" + f] = float(np.median(v))
        d["iqr_" + f] = float(np.percentile(v, 75.0) - np.percentile(v, 25.0))
    return d


def _stats(raws):
    st = {}
    for k in raws[0].keys():
        v = np.array([d[k] for d in raws], dtype=float)
        st[k] = (float(np.mean(v)), float(np.std(v)))
    return st


def _z(d, st, k):
    if k not in st or k not in d:
        return 0.0
    m, s = st[k]
    if s <= 1e-12:
        return 0.0
    return float((d[k] - m) / s)


def _row(d, st, spec):
    if spec == "const":
        return []
    if spec == "age":
        return [_z(d, st, "age")]
    if spec == "agesex":
        return [_z(d, st, "age"), _z(d, st, "sex")]
    if spec == "core3":
        return [_z(d, st, "age"), _z(d, st, "med_Shimmer:APQ11"), _z(d, st, "med_HNR")]
    if spec == "core5":
        return [_z(d, st, "age"), _z(d, st, "med_Shimmer:APQ11"), _z(d, st, "med_HNR"),
                _z(d, st, "med_DFA"), _z(d, st, "med_PPE")]
    jit = float(np.mean([_z(d, st, "med_" + f) for f in JIT]))
    shi = float(np.mean([_z(d, st, "med_" + f) for f in SHI]))
    noi = _z(d, st, "med_NHR") - _z(d, st, "med_HNR")
    non = float((_z(d, st, "med_RPDE") + _z(d, st, "med_PPE") - _z(d, st, "med_DFA")) / 3.0)
    if spec == "comp":
        return [_z(d, st, "age"), _z(d, st, "sex"), jit, shi, noi, non]
    var = float(np.mean([_z(d, st, "iqr_" + f) for f in SHI]))
    if spec == "compvar":
        return [_z(d, st, "age"), _z(d, st, "sex"), jit, shi, noi, non, var]
    return [_z(d, st, "age"), _z(d, st, "sex")] + [_z(d, st, "med_" + f) for f in VOICE]


def fit_predict(train, inputs, seed=0):
    findings = {"method": "patient_level_pooled_ridge",
                "data_use": "train X/y/groups for fitting; evaluation rows grouped by exact (age,sex) and aggregated -> explicit transductive use of unlabeled evaluation inputs",
                "fallback_used": False}
    inames = list(inputs["feature_names"])
    B = _logx(inputs["X"], inames)
    m = int(B.shape[0])
    try:
        names = list(train["feature_names"])
        A = _logx(train["X"], names)
        y = np.asarray(train["y"], dtype=float)
        groups = [str(g) for g in train["groups"]]
        ti = names.index("test_time")
        gmap = defaultdict(list)
        for i, g in enumerate(groups):
            gmap[g].append(i)
        gids = sorted(gmap.keys())
        G = len(gids)
        raws = [_praw(A, gmap[g], names) for g in gids]
        pmean = np.array([float(np.mean(y[gmap[g]])) for g in gids])
        prow_y = [y[gmap[g]] for g in gids]
        prow_dt = []
        slopes = []
        for g in gids:
            t = A[gmap[g], ti]
            prow_dt.append(t - float(np.mean(t)))
            if float(np.std(t)) > 1e-6:
                slopes.append(float(np.polyfit(t, y[gmap[g]], 1)[0]))
        gslope = float(np.median(slopes)) if slopes else 0.0
        TRENDS = [0.0, 0.5 * gslope, gslope]
        base = {}
        fold_mean = np.zeros(G)
        ref_sse = 0.0
        ref_n = 0
        ref_maes = []
        for k in range(G):
            tr = [j for j in range(G) if j != k]
            st = _stats([raws[j] for j in tr])
            ytr = pmean[tr]
            mu = float(np.mean(ytr))
            fold_mean[k] = mu
            med = float(np.median(np.concatenate([prow_y[j] for j in tr])))
            e = prow_y[k] - med
            ref_sse += float(np.sum(e * e))
            ref_n += int(e.shape[0])
            ref_maes.append(float(np.mean(np.abs(e))))
            for spec in SPECS:
                p = NFEAT[spec]
                if p == 0:
                    for a in ALPHAS:
                        base.setdefault((spec, a), np.zeros(G))[k] = mu
                    continue
                Xtr = np.array([_row(raws[j], st, spec) for j in tr], dtype=float).reshape(len(tr), p)
                xk = np.array(_row(raws[k], st, spec), dtype=float).reshape(1, p)
                for a in ALPHAS:
                    mdl = Ridge(alpha=a, fit_intercept=True).fit(Xtr, ytr)
                    base.setdefault((spec, a), np.zeros(G))[k] = float(mdl.predict(xk)[0])
        ref_rmse = math.sqrt(ref_sse / max(ref_n, 1))
        ref_gmae = float(np.mean(ref_maes))
        cands = []
        for spec in SPECS:
            for a in ALPHAS:
                if spec == "const" and a != ALPHAS[0]:
                    continue
                bp = base[(spec, a)]
                for w in SHRINKS:
                    for tc in TRENDS:
                        sse = 0.0
                        n = 0
                        maes = []
                        for k in range(G):
                            p = (1.0 - w) * bp[k] + w * fold_mean[k]
                            e = prow_y[k] - (p + tc * prow_dt[k])
                            sse += float(np.sum(e * e))
                            n += int(e.shape[0])
                            maes.append(float(np.mean(np.abs(e))))
                        cands.append({"spec": spec, "alpha": a, "w": w, "tc": tc,
                                      "rmse": math.sqrt(sse / max(n, 1)),
                                      "gmae": float(np.mean(maes))})
        ok = [c for c in cands if c["gmae"] <= ref_gmae * 1.05 and np.isfinite(c["rmse"])]
        if not ok:
            ok = [c for c in cands if np.isfinite(c["rmse"])]
        best = min(c["rmse"] for c in ok)
        band = [c for c in ok if c["rmse"] <= best * 1.005]
        band.sort(key=lambda c: (NFEAT[c["spec"]], -c["w"], abs(c["tc"]), c["rmse"]))
        sel = band[0]
        st = _stats(raws)
        mu_all = float(np.mean(pmean))
        mdl = None
        if NFEAT[sel["spec"]] > 0:
            Xall = np.array([_row(d, st, sel["spec"]) for d in raws], dtype=float)
            mdl = Ridge(alpha=sel["alpha"], fit_intercept=True).fit(Xall, pmean)
        rpp = float(np.median([len(gmap[g]) for g in gids]))
        ai = inames.index("age")
        si = inames.index("sex")
        tii = inames.index("test_time")
        kmap = defaultdict(list)
        for i in range(m):
            kmap[(round(float(B[i, ai]), 3), round(float(B[i, si]), 3))].append(i)
        egroups = []
        splits = 0
        for key in sorted(kmap.keys()):
            rows = kmap[key]
            if len(rows) > 1.8 * max(rpp, 1.0) and len(rows) >= 20:
                kk = int(max(2, round(len(rows) / max(rpp, 1.0))))
                cols = [inames.index(f) for f in VOICE]
                Z = B[np.array(rows), :][:, cols]
                zm = Z.mean(axis=0)
                zs = Z.std(axis=0)
                zs[zs < 1e-12] = 1.0
                lab = KMeans(n_clusters=kk, n_init=10, random_state=int(seed)).fit_predict((Z - zm) / zs)
                splits += 1
                for c in range(kk):
                    sub = [rows[j] for j in range(len(rows)) if int(lab[j]) == c]
                    if sub:
                        egroups.append(sub)
            else:
                egroups.append(rows)
        pred = np.full(m, mu_all, dtype=float)
        for rows in egroups:
            d = _praw(B, rows, inames)
            if mdl is not None:
                xg = np.array(_row(d, st, sel["spec"]), dtype=float).reshape(1, NFEAT[sel["spec"]])
                p = float(mdl.predict(xg)[0])
            else:
                p = mu_all
            p = (1.0 - sel["w"]) * p + sel["w"] * mu_all
            t = B[np.array(rows), tii]
            pred[np.array(rows)] = p + sel["tc"] * (t - float(np.mean(t)))
        lo = float(np.min(y)) - 2.0
        hi = float(np.max(y)) + 2.0
        pred = np.clip(pred, lo, hi)
        if not np.all(np.isfinite(pred)):
            raise ValueError("non_finite_prediction")
        top = sorted(ok, key=lambda c: c["rmse"])[:6]
        findings["selected"] = {"spec": sel["spec"], "alpha": sel["alpha"], "shrink": sel["w"],
                                "trend_coef": sel["tc"], "cv_row_rmse": round(sel["rmse"], 4),
                                "cv_group_mae": round(sel["gmae"], 4)}
        findings["leave_one_patient_out_reference_median"] = {"row_rmse": round(ref_rmse, 4), "group_mae": round(ref_gmae, 4)}
        findings["top_candidates"] = [{"spec": c["spec"], "alpha": c["alpha"], "w": c["w"],
                                       "tc": round(c["tc"], 6), "rmse": round(c["rmse"], 4),
                                       "gmae": round(c["gmae"], 4)} for c in top]
        findings["train_within_patient_median_slope"] = round(gslope, 6)
        findings["train_patients"] = G
        findings["eval_groups"] = len(egroups)
        findings["eval_group_sizes"] = sorted([len(r) for r in egroups])[:40]
        findings["eval_kmeans_subsplits"] = splits
        findings["note"] = "selection uses leave-one-patient-out row RMSE on train only; no evaluation labels used; internal winner may differ after refit on train+development"
        return {"prediction": [float(v) for v in pred], "findings": findings}
    except Exception as exc:
        findings["fallback_used"] = True
        findings["fallback_reason"] = type(exc).__name__ + ": " + str(exc)[:200]
        findings["fallback_rule"] = "constant train median prediction; intended patient-level pooled ridge did NOT run"
        yv = np.asarray(train["y"], dtype=float)
        med = float(np.median(yv)) if yv.size else 0.0
        return {"prediction": [med] * m, "findings": findings}
