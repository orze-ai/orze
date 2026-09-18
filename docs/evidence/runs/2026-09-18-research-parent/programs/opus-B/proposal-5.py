import numpy as np

SHRINK = 0.5
DRIFT_CLIP = 2.5


def _pat_arrays(y, g):
    keys = np.unique(g)
    pm = np.array([float(y[g == k].mean()) for k in keys], dtype=float)
    pn = np.array([float((g == k).sum()) for k in keys], dtype=float)
    pv = np.array([float(y[g == k].var()) for k in keys], dtype=float)
    return keys, pm, pn, pv


def _rmse_opt_const(pm, pn, pv, k_draw, rng, n_sim, res):
    """Constant minimising the mean simulated cohort ROW-RMSE over draws of
    k_draw patients (without replacement) from the supplied patients."""
    m = int(len(pm))
    k = int(min(k_draw, m))
    sm = np.empty(int(n_sim), dtype=float)
    sv = np.empty(int(n_sim), dtype=float)
    for i in range(int(n_sim)):
        idx = rng.choice(m, size=k, replace=False)
        w = pn[idx].astype(float)
        w = w / w.sum()
        mu = float(np.dot(w, pm[idx]))
        sm[i] = mu
        sv[i] = float(np.dot(w, pv[idx] + (pm[idx] - mu) ** 2))
    lo = float(pm.min()) - 2.0
    hi = float(pm.max()) + 2.0
    grid = np.linspace(lo, hi, int(res))
    obj = np.empty(grid.size, dtype=float)
    for j in range(grid.size):
        obj[j] = float(np.mean(np.sqrt(sv + (grid[j] - sm) ** 2)))
    j = int(np.argmin(obj))
    return float(grid[j]), float(obj[j])


def _within_slope(t, y, g):
    num = 0.0
    den = 0.0
    for k in np.unique(g):
        m = g == k
        if int(m.sum()) < 5:
            continue
        tt = t[m] - float(t[m].mean())
        yy = y[m] - float(y[m].mean())
        num += float(np.dot(tt, yy))
        den += float(np.dot(tt, tt))
    if den <= 0.0:
        return 0.0
    return float(num / den)


def _rmse(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(float(np.mean(d * d))))


def fit_predict(train, inputs, seed):
    findings = {
        "design": "pre-declared rule: constant = argmin_c mean_draws sqrt(cohort row MSE) over simulated 7-patient cohorts drawn from the 28 train patients, plus a half-shrunk clipped mean-zero global test_time drift term; internal held-out-7 comparisons are reported only, never used to switch the output",
        "transductive": "drift term is centred on the median test_time of the unlabeled evaluation inputs; no evaluation labels used",
        "fallback_used": False,
        "errors": [],
    }
    Xe = np.asarray(inputs["X"], dtype=float)
    n_eval = int(Xe.shape[0])
    try:
        X = np.asarray(train["X"], dtype=float)
        y = np.asarray(train["y"], dtype=float)
        g = np.asarray(train["groups"])
        names = [str(v) for v in train["feature_names"]]
        ti = names.index("test_time")
        t_tr = X[:, ti].astype(float)
        t_ev = Xe[:, ti].astype(float)

        keys, pm, pn, pv = _pat_arrays(y, g)
        rng = np.random.default_rng(1729)

        c_star, obj_star = _rmse_opt_const(pm, pn, pv, 7, rng, 8000, 1201)
        c_row_mean = float(np.mean(y))
        c_row_med = float(np.median(y))
        c_pm_mean = float(np.mean(pm))
        c_pm_med = float(np.median(pm))

        b = _within_slope(t_tr, y, g)
        centre = float(np.median(t_ev))
        drift = np.clip(SHRINK * b * (t_ev - centre), -DRIFT_CLIP, DRIFT_CLIP)
        pred = c_star + drift

        findings["constants_train_only"] = {
            "simulated_rmse_optimal": round(c_star, 4),
            "simulated_objective": round(obj_star, 4),
            "row_mean": round(c_row_mean, 4),
            "row_median": round(c_row_med, 4),
            "patient_mean_mean": round(c_pm_mean, 4),
            "patient_mean_median": round(c_pm_med, 4),
            "n_train_patients": int(len(keys)),
        }
        findings["drift"] = {
            "pooled_within_slope_upd_per_day": round(b, 6),
            "shrink": SHRINK,
            "eval_test_time_median": round(centre, 4),
            "drift_sd": round(float(np.std(drift)), 4),
            "drift_min": round(float(np.min(drift)), 4),
            "drift_max": round(float(np.max(drift)), 4),
        }

        # internal patient-held-out validation matched to the 7-patient cohort structure
        rng2 = np.random.default_rng(20260918)
        m = int(len(keys))
        k_hold = int(min(7, max(1, m - 4)))
        variants = ["row_mean", "row_median", "patient_mean_median", "sim_opt", "sim_opt_plus_drift"]
        acc = {v: [] for v in variants}
        reps = 120
        for _ in range(reps):
            hold = rng2.choice(m, size=k_hold, replace=False)
            hold_keys = set(keys[hold].tolist())
            mask_te = np.array([kk in hold_keys for kk in g.tolist()])
            mask_tr = ~mask_te
            if int(mask_tr.sum()) < 50 or int(mask_te.sum()) < 20:
                continue
            y_tr = y[mask_tr]
            g_tr = g[mask_tr]
            t_sub = t_tr[mask_tr]
            y_te = y[mask_te]
            t_te = t_tr[mask_te]
            k2, pm2, pn2, pv2 = _pat_arrays(y_tr, g_tr)
            c_sim, _ = _rmse_opt_const(pm2, pn2, pv2, k_hold, rng2, 1200, 401)
            b2 = _within_slope(t_sub, y_tr, g_tr)
            d2 = np.clip(SHRINK * b2 * (t_te - float(np.median(t_te))), -DRIFT_CLIP, DRIFT_CLIP)
            acc["row_mean"].append(_rmse(y_te, float(np.mean(y_tr))))
            acc["row_median"].append(_rmse(y_te, float(np.median(y_tr))))
            acc["patient_mean_median"].append(_rmse(y_te, float(np.median(pm2))))
            acc["sim_opt"].append(_rmse(y_te, c_sim))
            acc["sim_opt_plus_drift"].append(_rmse(y_te, c_sim + d2))
        summary = {}
        base = np.asarray(acc["row_median"], dtype=float)
        for v in variants:
            a = np.asarray(acc[v], dtype=float)
            if a.size == 0:
                summary[v] = None
                continue
            entry = {
                "mean_rmse": round(float(np.mean(a)), 4),
                "median_rmse": round(float(np.median(a)), 4),
                "n_cohorts": int(a.size),
            }
            if base.size == a.size and base.size > 0:
                entry["frac_better_than_row_median"] = round(float(np.mean(a < base)), 4)
            summary[v] = entry
        findings["internal_heldout_patient_cohorts"] = summary
        findings["caveat"] = "internal cohorts reuse the same 28 train patients, so these comparisons are correlated and optimistic; idea-7ea121 showed 28-patient internal selection can pick the wrong constant, hence no selection was performed here"
        findings["prediction_summary"] = {
            "n": n_eval,
            "mean": round(float(np.mean(pred)), 4),
            "sd": round(float(np.std(pred)), 4),
            "min": round(float(np.min(pred)), 4),
            "max": round(float(np.max(pred)), 4),
        }
        return {"prediction": [float(v) for v in pred], "findings": findings}
    except Exception as e:  # broad handler: must announce the fallback explicitly
        findings["fallback_used"] = True
        findings["errors"].append(repr(e)[:300])
        c = float(np.median(np.asarray(train["y"], dtype=float)))
        findings["fallback"] = {"rule": "train row median constant", "c": round(c, 4)}
        return {"prediction": [c] * n_eval, "findings": findings}
