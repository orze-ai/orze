import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HGB_KW = {"max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 63,
          "l2_regularization": 1.0, "min_samples_leaf": 20}

CANDS = [
    {"name": "direct_ed", "use_ed": True, "trend": False, "damp": 0.0, "cap": 0.0},
    {"name": "no_trend", "use_ed": False, "trend": False, "damp": 0.0, "cap": 0.0},
    {"name": "trend_freeze", "use_ed": False, "trend": True, "damp": 1.0, "cap": 0.0},
    {"name": "trend_damp", "use_ed": False, "trend": True, "damp": 0.5, "cap": 400.0},
    {"name": "trend_full", "use_ed": False, "trend": True, "damp": 1.0, "cap": 400.0},
]


def _build(X, names, use_ed):
    X = np.asarray(X, dtype=float)
    ix = {}
    for i, n in enumerate(names):
        ix[n] = i
    hr = X[:, ix["hr"]]
    ed = X[:, ix["elapsed_day"]]
    wk = X[:, ix["workingday"]]
    doy = np.mod(ed, 365.25)
    extra = [np.sin(2.0 * np.pi * hr / 24.0),
             np.cos(2.0 * np.pi * hr / 24.0),
             np.sin(4.0 * np.pi * hr / 24.0),
             np.cos(4.0 * np.pi * hr / 24.0),
             np.sin(2.0 * np.pi * doy / 365.25),
             np.cos(2.0 * np.pi * doy / 365.25),
             hr + 24.0 * wk,
             X[:, ix["temp"]] * X[:, ix["hum"]],
             X[:, ix["atemp"]] - X[:, ix["temp"]]]
    keep = [i for i, n in enumerate(names) if (use_ed or n != "elapsed_day")]
    return np.column_stack([X[:, keep]] + extra), ed


def _offset(ed, a, b, ed_max, cfg):
    ed_c = np.minimum(ed, ed_max + float(cfg["cap"]))
    inside = np.minimum(ed_c, ed_max)
    outside = np.maximum(ed_c - ed_max, 0.0)
    return a + b * inside + b * float(cfg["damp"]) * outside


def _fit(Xr, y, g, names, cfg, seed):
    Xf, ed = _build(Xr, names, cfg["use_ed"])
    t = np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None))
    rs = 1729 + int(abs(int(seed)) % 90000)
    m = HistGradientBoostingRegressor(random_state=rs, **HGB_KW)
    m.fit(Xf, t)
    fit = np.asarray(m.predict(Xf), dtype=float)
    ed_max = float(np.max(ed))
    a = 0.0
    b = 0.0
    n_dates = 0
    if cfg["trend"]:
        gg = np.asarray([str(v) for v in g])
        uq, inv = np.unique(gg, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        cnts = np.bincount(inv).astype(float)
        dm = np.bincount(inv, weights=(t - fit)) / cnts
        de = np.bincount(inv, weights=ed) / cnts
        n_dates = int(uq.shape[0])
        if n_dates >= 3 and float(np.ptp(de)) > 0.0:
            A = np.column_stack([np.ones_like(de), de])
            coef, _r, _rk, _sv = np.linalg.lstsq(A, dm, rcond=None)
            if np.all(np.isfinite(coef)):
                a = float(coef[0])
                b = float(coef[1])
        fit = fit + _offset(ed, a, b, ed_max, cfg)
    res = t - fit
    sm = float(np.mean(np.exp(np.clip(res, -20.0, 20.0))))
    if (not np.isfinite(sm)) or sm <= 0.0:
        sm = 1.0
    return {"m": m, "a": a, "b": b, "ed_max": ed_max, "smear": sm,
            "cfg": cfg, "trend_dates": n_dates}


def _pred(mod, Xr, names):
    cfg = mod["cfg"]
    Xf, ed = _build(Xr, names, cfg["use_ed"])
    p = np.asarray(mod["m"].predict(Xf), dtype=float)
    if cfg["trend"]:
        p = p + _offset(ed, mod["a"], mod["b"], mod["ed_max"], cfg)
    p = np.exp(np.clip(p, -20.0, 20.0)) * float(mod["smear"]) - 1.0
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, None)


def fit_predict(train, inputs, seed):
    findings = {"candidates": [], "fallback_used": False, "notes": [],
                "transductive_use": False,
                "selection_rule": "mean_rmse_over_time_blocked_folds_incl_long_horizon"}
    Xtr = np.asarray(train["X"], dtype=float)
    y = np.asarray(train["y"], dtype=float)
    names = list(train["feature_names"])
    Xev = np.asarray(inputs["X"], dtype=float)
    ev_names = list(inputs["feature_names"])
    try:
        if ev_names != names:
            findings["notes"].append("feature_name_order_differs_used_each_own_index")
        gg = np.asarray([str(v) for v in train["groups"]])
        ied = names.index("elapsed_day")
        edcol = Xtr[:, ied]
        uq = np.unique(gg).tolist()
        key = {}
        for d in uq:
            key[d] = float(np.mean(edcol[gg == d]))
        order = sorted(uq, key=lambda d: key[d])
        nd = len(order)
        findings["n_train_dates"] = nd
        specs = []
        if nd >= 150:
            specs.append(("h1_last61", order[:nd - 61], order[nd - 61:]))
        if nd >= 250:
            specs.append(("h1_prev61", order[:nd - 122], order[nd - 122:nd - 61]))
        if nd >= 320:
            specs.append(("h1_prev2_61", order[:nd - 183], order[nd - 183:nd - 122]))
        if nd >= 250:
            specs.append(("h2_long61", order[:nd - 122], order[nd - 61:]))
        findings["folds"] = [s[0] for s in specs]
        if not specs:
            findings["notes"].append("too_few_dates_for_time_blocked_validation")
        scores = {}
        for cfg in CANDS:
            rec = {"name": cfg["name"], "per_fold": {}}
            try:
                for fname, trd, vad in specs:
                    tr_m = np.isin(gg, np.asarray(trd))
                    va_m = np.isin(gg, np.asarray(vad))
                    mod = _fit(Xtr[tr_m], y[tr_m], gg[tr_m], names, cfg, seed)
                    p = _pred(mod, Xtr[va_m], names)
                    err = p - y[va_m]
                    rec["per_fold"][fname] = {
                        "rmse": float(np.sqrt(np.mean(err ** 2))),
                        "mae": float(np.mean(np.abs(err))),
                        "slope": float(mod["b"])}
                if rec["per_fold"]:
                    vals = [v["rmse"] for v in rec["per_fold"].values()]
                    rec["mean_rmse"] = float(np.mean(vals))
                    rec["mean_mae"] = float(np.mean([v["mae"] for v in rec["per_fold"].values()]))
                    scores[cfg["name"]] = rec["mean_rmse"]
            except Exception as exc:
                rec["error"] = type(exc).__name__
            findings["candidates"].append(rec)
        if scores:
            best_name = min(sorted(scores.keys()), key=lambda k: scores[k])
        else:
            best_name = "direct_ed"
            findings["notes"].append("no_valid_candidate_score_used_direct_ed_branch")
        best = [c for c in CANDS if c["name"] == best_name][0]
        findings["selected_branch"] = best_name
        findings["selected_internal_mean_rmse"] = scores.get(best_name)
        findings["direct_ed_internal_mean_rmse"] = scores.get("direct_ed")
        final = _fit(Xtr, y, gg, names, best, seed)
        findings["final_trend_intercept"] = float(final["a"])
        findings["final_trend_slope_per_day"] = float(final["b"])
        findings["final_ed_max_train"] = float(final["ed_max"])
        findings["final_smear"] = float(final["smear"])
        Xf_chk, _e = _build(Xtr, names, best["use_ed"])
        Xf_ev, ed_ev = _build(Xev, ev_names, best["use_ed"])
        if Xf_ev.shape[1] != Xf_chk.shape[1]:
            raise ValueError("feature_width_mismatch")
        findings["eval_ed_min"] = float(np.min(ed_ev))
        findings["eval_ed_max"] = float(np.max(ed_ev))
        pred = _pred(final, Xev, ev_names)
        findings["pred_rows"] = int(pred.shape[0])
        findings["pred_mean"] = float(np.mean(pred))
        findings["pred_max"] = float(np.max(pred))
        findings["train_y_mean"] = float(np.mean(y))
        return {"prediction": pred.tolist(), "findings": findings}
    except Exception as exc:
        findings["fallback_used"] = True
        findings["fallback_reason"] = type(exc).__name__
        findings["notes"].append("broad_handler_reference_hgb_250_0.08_31_on_raw_columns")
        fb = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                           max_leaf_nodes=31,
                                           l2_regularization=1,
                                           random_state=1729)
        fb.fit(Xtr, y)
        p = np.asarray(fb.predict(Xev), dtype=float)
        p = np.clip(np.where(np.isfinite(p), p, float(np.mean(y))), 0.0, None)
        return {"prediction": p.tolist(), "findings": findings}
