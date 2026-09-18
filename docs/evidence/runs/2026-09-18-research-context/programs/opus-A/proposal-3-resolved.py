import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

REF = {"max_iter": 250, "learning_rate": 0.08, "max_leaf_nodes": 31,
       "l2_regularization": 1, "random_state": 1729}


def _trend(el, y):
    days = np.round(el).astype(int)
    ud = np.unique(days)
    m = np.empty(ud.size, dtype=float)
    for i in range(ud.size):
        m[i] = y[days == ud[i]].mean()
    lm = np.log(np.maximum(m, 1.0))
    t = ud.astype(float)
    cols = [np.ones_like(t), t]
    for h in (1.0, 2.0):
        a = 2.0 * np.pi * h * t / 365.25
        cols.append(np.sin(a))
        cols.append(np.cos(a))
    A = np.vstack(cols).T
    coef = np.linalg.lstsq(A, lm, rcond=None)[0]
    return float(coef[1]), float(t.max()), float(t.mean())


def _g(el, b, dmax, dref, kappa):
    eff = np.where(el <= dmax, el, dmax + kappa * (el - dmax))
    return np.exp(b * (eff - dref))


def _fit_base(Xtr, ytr):
    mdl = HistGradientBoostingRegressor(**REF)
    mdl.fit(Xtr, ytr)
    return mdl


def _pred_base(mdl, Xev):
    return np.maximum(mdl.predict(Xev), 0.0)


def _fit_trendmodel(Xtr, ytr, eltr, kappa):
    b, dmax, dref = _trend(eltr, ytr)
    g = _g(eltr, b, dmax, dref, kappa)
    g = np.maximum(g, 1e-6)
    r = ytr / g
    w = g * g
    mdl = HistGradientBoostingRegressor(**REF)
    mdl.fit(Xtr, r, sample_weight=w)
    return (mdl, b, dmax, dref, kappa)


def _pred_trendmodel(st, Xev, elev):
    mdl, b, dmax, dref, kappa = st
    g = _g(elev, b, dmax, dref, kappa)
    return np.maximum(mdl.predict(Xev) * g, 0.0)


def _rmse(a, p):
    d = np.asarray(p, dtype=float) - np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def fit_predict(train, inputs, seed):
    names = list(train['feature_names'])
    Xtr_full = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xev_full = np.asarray(inputs['X'], dtype=float)
    ei = names.index('elapsed_day')
    el_tr = Xtr_full[:, ei]
    el_ev = Xev_full[:, ei]
    keep = [j for j in range(len(names)) if j != ei]
    Xtr_ne = Xtr_full[:, keep]
    Xev_ne = Xev_full[:, keep]
    day_tr = np.round(el_tr).astype(int)
    dates = np.unique(day_tr)
    n = int(dates.size)
    findings = {
        "fallback_used": False,
        "n_train_rows": int(Xtr_full.shape[0]),
        "n_train_dates": n,
        "n_pred_rows": int(Xev_full.shape[0]),
        "elapsed_train_max": float(el_tr.max()),
        "elapsed_inputs_min": float(el_ev.min()),
        "elapsed_inputs_max": float(el_ev.max()),
        "inputs_beyond_train_elapsed_fraction": float(np.mean(el_ev > el_tr.max())),
        "transductive_use": "none_for_fitting_or_selection; unlabeled input elapsed_day used only to evaluate the level factor at prediction time and for range diagnostics",
    }
    try:
        b_all, dmax_all, dref_all = _trend(el_tr, y)
        findings["full_train_log_slope_per_day"] = round(b_all, 6)
        findings["full_train_implied_annual_growth"] = round(float(np.exp(b_all * 365.25)), 4)
    except Exception as exc:
        findings["trend_diag_error"] = type(exc).__name__
    block = 40
    nf = 4
    fold_res = {"baseline": [], "trend_full": [], "trend_frozen": []}
    ok = True
    try:
        for k in range(nf):
            c = n - (nf - k) * block
            if c < 200:
                ok = False
                break
            trd = set(dates[:c].tolist())
            vad = set(dates[c:c + block].tolist())
            mtr = np.array([d in trd for d in day_tr], dtype=bool)
            mva = np.array([d in vad for d in day_tr], dtype=bool)
            if mtr.sum() < 500 or mva.sum() < 100:
                ok = False
                break
            b0 = _fit_base(Xtr_full[mtr], y[mtr])
            fold_res["baseline"].append(round(_rmse(y[mva], _pred_base(b0, Xtr_full[mva])), 3))
            for nm, kap in (("trend_full", 1.0), ("trend_frozen", 0.0)):
                st = _fit_trendmodel(Xtr_ne[mtr], y[mtr], el_tr[mtr], kap)
                fold_res[nm].append(round(_rmse(y[mva], _pred_trendmodel(st, Xtr_ne[mva], el_tr[mva])), 3))
    except Exception as exc:
        ok = False
        findings["fold_error"] = type(exc).__name__
        findings["fallback_used"] = True
    findings["fold_rmse"] = fold_res
    findings["selection_rule"] = ("four strictly chronological 40-date holdout blocks taken from training dates only; "
                                  "a challenger replaces the fixed baseline only if its mean RMSE is lower AND it wins in >=3 of 4 folds")
    sel = "baseline"
    if ok and len(fold_res["baseline"]) == nf:
        bm = float(np.mean(fold_res["baseline"]))
        findings["baseline_mean_rmse"] = round(bm, 3)
        summary = {}
        best = None
        for nm in ("trend_full", "trend_frozen"):
            mm = float(np.mean(fold_res[nm]))
            wins = int(sum(1 for i in range(nf) if fold_res[nm][i] < fold_res["baseline"][i]))
            summary[nm] = {"mean_rmse": round(mm, 3), "wins_vs_baseline": wins}
            if mm < bm and wins >= 3 and (best is None or mm < best[1]):
                best = (nm, mm)
        findings["challenger_summary"] = summary
        findings["old_rule_would_select"] = (best[0] if best is not None else "baseline")
    else:
        findings["selection_note"] = "chronological fold diagnostics unavailable; the returned prediction is still the unconditional trend_frozen model, only the diagnostic folds are missing"
        findings["fallback_used"] = True
    sel = "trend_frozen"
    findings["branch_policy"] = "unconditional trend_frozen (kappa=0) this round: detrended ratio target with level frozen at the last training day; no data-driven branch switch, so the development score measures this representation directly"
    findings["selected_branch"] = sel
    try:
        if sel == "baseline":
            mdl = _fit_base(Xtr_full, y)
            pred = _pred_base(mdl, Xev_full)
            findings["final_fit"] = "HistGradientBoosting(reference settings) on all raw columns, raw target"
        else:
            kap = 1.0 if sel == "trend_full" else 0.0
            st = _fit_trendmodel(Xtr_ne, y, el_tr, kap)
            pred = _pred_trendmodel(st, Xev_ne, el_ev)
            findings["final_fit"] = ("HistGradientBoosting(reference settings) on ratio target y/g(d) with weights g(d)^2, "
                                     "elapsed_day removed from features, level factor g from log-linear daily trend, kappa=%s" % kap)
            findings["selected_log_slope_per_day"] = round(float(st[1]), 6)
            findings["selected_kappa"] = float(kap)
            try:
                bmdl = _fit_base(Xtr_full, y)
                bpred = _pred_base(bmdl, Xev_full)
                findings["reference_baseline_pred_mean"] = round(float(bpred.mean()), 3)
                findings["pred_mean_ratio_vs_reference_baseline"] = round(float(pred.mean() / max(float(bpred.mean()), 1e-9)), 4)
                findings["mean_abs_diff_vs_reference_baseline"] = round(float(np.mean(np.abs(pred - bpred))), 3)
                findings["reference_baseline_role"] = "diagnostic_only_not_used_in_returned_prediction"
            except Exception as exc2:
                findings["ref_diag_error"] = type(exc2).__name__
    except Exception as exc:
        findings["final_fit_error"] = type(exc).__name__
        findings["fallback_used"] = True
        findings["selected_branch"] = "baseline_after_final_fit_error"
        mdl = _fit_base(Xtr_full, y)
        pred = _pred_base(mdl, Xev_full)
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, float(np.median(y)))
    pred = np.maximum(pred, 0.0)
    findings["pred_mean"] = round(float(pred.mean()), 3)
    findings["pred_max"] = round(float(pred.max()), 3)
    findings["train_y_mean"] = round(float(y.mean()), 3)
    return {"prediction": pred.tolist(), "findings": findings}
