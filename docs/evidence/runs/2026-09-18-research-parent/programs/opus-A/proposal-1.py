import json, math
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr


def _f(v, nd=3):
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return round(x, nd)


def _get(d):
    X = np.asarray(d["X"], dtype=float)
    y = np.asarray(d["y"], dtype=float)
    g = d.get("groups")
    if g is None:
        g = np.array(["all"] * len(y))
    else:
        g = np.array([str(v) for v in g])
    return X, y, g


def _metrics(y, p, g):
    d = np.asarray(y, float) - np.asarray(p, float)
    per = [float(np.mean(np.abs(d[g == u]))) for u in np.unique(g)]
    return {"rmse": _f(math.sqrt(float(np.mean(d ** 2)))),
            "mae": _f(float(np.mean(np.abs(d)))),
            "gmae": _f(float(np.mean(per)))}


def _zfit(A):
    mu = A.mean(0)
    sd = A.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def _pool(p, lab):
    o = np.asarray(p, float).copy()
    for u in np.unique(lab):
        m = (lab == u)
        o[m] = float(o[m].mean())
    return o


def _cluster(X, k, tt_idx):
    Z = np.delete(X, [tt_idx], axis=1) if tt_idx is not None else X.copy()
    mu, sd = _zfit(Z)
    Z = (Z - mu) / sd
    n = Z.shape[0]
    k = int(max(1, min(int(k), n)))
    if k == 1:
        return np.zeros(n, dtype=int)
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)


def _ridge(a):
    def b(Xa, ya, Xb):
        mu, sd = _zfit(Xa)
        m = Ridge(alpha=a).fit((Xa - mu) / sd, ya)
        return m.predict((Xb - mu) / sd)
    return b


def _hgb(Xa, ya, Xb):
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                      max_leaf_nodes=31, l2_regularization=1.0,
                                      random_state=1729)
    m.fit(Xa, ya)
    return m.predict(Xb)


def _const(Xa, ya, Xb):
    return np.full(Xb.shape[0], float(np.mean(ya)))


def analyze(data, history, seed=1729):
    res = {"scope": "analysis only; no prediction returned",
           "data_use": "train and development X/y/groups supplied to analyze; clustering of feature rows without labels is transductive and reported as such"}
    try:
        tr = data["train"]
        dv = data["development"]
        fn = [str(s) for s in tr["feature_names"]]
        Xtr, ytr, gtr = _get(tr)
        Xdv, ydv, gdv = _get(dv)
        tt = fn.index("test_time") if "test_time" in fn else None
        res["shape"] = {"train_rows": int(Xtr.shape[0]), "dev_rows": int(Xdv.shape[0]),
                        "train_patients": int(len(np.unique(gtr))),
                        "dev_patients": int(len(np.unique(gdv))),
                        "n_features": len(fn), "test_time_index": tt}
    except Exception as e:
        return {"fatal_setup_error": str(e)[:300]}

    try:
        def decomp(y, g):
            us = np.unique(g)
            pm = np.array([float(y[g == u].mean()) for u in us])
            wv = float(np.mean([float(np.var(y[g == u])) for u in us]))
            return {"between_sd": _f(float(np.std(pm))),
                    "within_sd": _f(math.sqrt(max(wv, 0.0))),
                    "total_sd": _f(float(np.std(y))),
                    "pm_min": _f(float(pm.min())), "pm_max": _f(float(pm.max())),
                    "rows_per_patient": _f(float(len(y)) / max(1, len(us)), 1)}
        res["y_variance"] = {"train": decomp(ytr, gtr), "dev": decomp(ydv, gdv)}
    except Exception as e:
        res["y_variance_error"] = str(e)[:200]

    try:
        us = np.unique(gtr)
        idx = {u: k for k, u in enumerate(us)}
        pos = np.array([idx[u] for u in gtr])
        pm_y = np.array([float(ytr[gtr == u].mean()) for u in us])
        feats = []
        for j, name in enumerate(fn):
            col = Xtr[:, j]
            pmf = np.array([float(col[gtr == u].mean()) for u in us])
            tvar = float(np.var(col))
            icc = _f(float(np.var(pmf)) / tvar) if tvar > 1e-15 else None
            rb = rhob = None
            if float(np.std(pmf)) > 1e-12:
                rb = _f(float(np.corrcoef(pmf, pm_y)[0, 1]))
                try:
                    rhob = _f(float(spearmanr(pmf, pm_y).correlation))
                except Exception:
                    rhob = None
            cc = col - pmf[pos]
            cy = ytr - pm_y[pos]
            rw = None
            if float(np.std(cc)) > 1e-12 and float(np.std(cy)) > 1e-12:
                rw = _f(float(np.corrcoef(cc, cy)[0, 1]))
            feats.append({"f": name, "icc": icc, "rb": rb, "rhob": rhob, "rw": rw})
        res["features"] = feats
    except Exception as e:
        res["features_error"] = str(e)[:200]

    try:
        if tt is not None:
            us = np.unique(gtr)
            sl, r2 = [], []
            pm_y = []
            for u in us:
                m = (gtr == u)
                t = Xtr[m, tt]
                yy = ytr[m]
                pm_y.append(float(yy.mean()))
                if len(yy) > 2 and float(np.std(t)) > 1e-9:
                    A = np.vstack([t, np.ones_like(t)]).T
                    coef, _, _, _ = np.linalg.lstsq(A, yy, rcond=None)
                    pred = A.dot(coef)
                    ss = float(np.sum((yy - yy.mean()) ** 2))
                    sl.append(float(coef[0]))
                    r2.append(float(1.0 - np.sum((yy - pred) ** 2) / ss) if ss > 1e-12 else None)
            sl_a = np.array(sl, float)
            r2_a = np.array([v for v in r2 if v is not None], float)
            res["within_patient_time_trend"] = {
                "slope_mean": _f(float(sl_a.mean())) if sl_a.size else None,
                "slope_sd": _f(float(sl_a.std())) if sl_a.size else None,
                "slope_pos_frac": _f(float(np.mean(sl_a > 0))) if sl_a.size else None,
                "linear_r2_mean": _f(float(r2_a.mean())) if r2_a.size else None,
                "corr_slope_patientmean": _f(float(np.corrcoef(sl_a, np.array(pm_y[:sl_a.size]))[0, 1])) if sl_a.size > 2 else None}
    except Exception as e:
        res["time_trend_error"] = str(e)[:200]

    builders = {"const": _const, "ridge1": _ridge(1.0), "ridge10": _ridge(10.0),
                "ridge100": _ridge(100.0), "ridge1000": _ridge(1000.0), "hgb250": _hgb}

    try:
        us = np.unique(gtr)
        gk = GroupKFold(n_splits=int(min(7, len(us))))
        splits = list(gk.split(Xtr, ytr, gtr))
        cv = {}
        oof_store = {}
        for nm, b in builders.items():
            p = np.zeros(len(ytr))
            for a, t in splits:
                p[t] = b(Xtr[a], ytr[a], Xtr[t])
            oof_store[nm] = p
            cv[nm] = _metrics(ytr, p, gtr)
        res["patient_held_out_cv_train"] = cv
        pool = {}
        for nm in ["ridge10", "ridge100", "hgb250"]:
            if nm not in oof_store:
                continue
            po = np.zeros(len(ytr))
            pc = np.zeros(len(ytr))
            for a, t in splits:
                seg = oof_store[nm][t]
                po[t] = _pool(seg, gtr[t])
                pc[t] = _pool(seg, _cluster(Xtr[t], len(np.unique(gtr[t])), tt))
            pool[nm] = {"oracle_patient_pool": _metrics(ytr, po, gtr),
                        "cluster_pool_true_k": _metrics(ytr, pc, gtr)}
        res["pooling_effect_cv"] = pool
    except Exception as e:
        res["cv_error"] = str(e)[:200]

    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=int(seed) % 100000)
        row_cv = {}
        for nm in ["ridge10", "hgb250"]:
            p = np.zeros(len(ytr))
            for a, t in kf.split(Xtr):
                p[t] = builders[nm](Xtr[a], ytr[a], Xtr[t])
            row_cv[nm] = _metrics(ytr, p, gtr)
        res["row_held_out_cv_train"] = row_cv
        res["optimism_note"] = "compare row_held_out_cv_train with patient_held_out_cv_train; a large gap indicates within-patient leakage rather than transferable signal"
    except Exception as e:
        res["row_cv_error"] = str(e)[:200]

    try:
        dev = {}
        dev_pred = {}
        for nm, b in builders.items():
            p = b(Xtr, ytr, Xdv)
            dev_pred[nm] = p
            dev[nm] = _metrics(ydv, p, gdv)
        res["dev_fit_on_train"] = dev
        dpool = {}
        for nm in ["ridge10", "ridge100", "hgb250"]:
            if nm not in dev_pred:
                continue
            e = {"oracle_patient_pool": _metrics(ydv, _pool(dev_pred[nm], gdv), gdv)}
            for k in [4, 7, 12, 20]:
                lab = _cluster(Xdv, k, tt)
                e["cluster_pool_k%d" % k] = _metrics(ydv, _pool(dev_pred[nm], lab), gdv)
            for w in [0.25, 0.5, 0.75]:
                base = float(np.mean(ytr))
                shr = w * np.asarray(dev_pred[nm], float) + (1.0 - w) * base
                e["shrink_to_train_mean_w%.2f" % w] = _metrics(ydv, shr, gdv)
            dpool[nm] = e
        res["dev_pooling_and_shrinkage"] = dpool
    except Exception as e:
        res["dev_error"] = str(e)[:200]

    try:
        cq = {}
        for k in [4, 7, 12, 20]:
            lab = _cluster(Xdv, k, tt)
            pur = 0.0
            for u in np.unique(lab):
                m = (lab == u)
                vals, cnt = np.unique(gdv[m], return_counts=True)
                pur += float(cnt.max())
            cq["k%d" % k] = {"ari": _f(float(adjusted_rand_score(gdv, lab))),
                             "purity": _f(pur / float(len(gdv)))}
        res["dev_transductive_cluster_quality"] = cq
        ages = Xdv[:, fn.index("age")] if "age" in fn else None
        sexes = Xdv[:, fn.index("sex")] if "sex" in fn else None
        if ages is not None and sexes is not None:
            combos = set(zip(ages.tolist(), sexes.tolist()))
            res["dev_age_sex_unique_combos"] = {"combos": int(len(combos)),
                                                "patients": int(len(np.unique(gdv)))}
        res["cluster_caveat"] = "true patient count is unknown at prediction time; k sensitivity above indicates whether a pooling wrapper is robust"
    except Exception as e:
        res["cluster_error"] = str(e)[:200]

    try:
        hs = []
        for h in (history or []):
            if not isinstance(h, dict):
                continue
            fa = h.get("facts") or {}
            hs.append({"task": str(h.get("task_id"))[:32], "kind": str((h.get("action") or {}).get("kind"))[:12],
                       "valid": bool(h.get("valid")), "loss": _f(fa.get("loss")), "gmae": _f(fa.get("group_mae"))}) 
        res["history_seen"] = hs[-8:]
    except Exception as e:
        res["history_error"] = str(e)[:200]

    try:
        s = json.dumps(res)
        if len(s.encode("utf-8")) > 7800 and "features" in res:
            fl = res["features"]
            keep = sorted(fl, key=lambda d: -abs(d.get("rb") or 0.0))[:8]
            res["features"] = keep
            res["features_trimmed"] = True
        s = json.dumps(res)
        if len(s.encode("utf-8")) > 8000:
            res.pop("history_seen", None)
            res.pop("features", None)
            res["trimmed"] = "dropped detail blocks to respect 8192-byte findings limit"
    except Exception as e:
        res["size_error"] = str(e)[:200]
    return res
