import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

BASE_KW = {"max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 63,
           "l2_regularization": 1.0, "min_samples_leaf": 20}

CANDS = [
    {"name": "log_smear_1seed", "obj": "log", "seeds": 1, "hl": 0.0},
    {"name": "log_smear_3seed", "obj": "log", "seeds": 3, "hl": 0.0},
    {"name": "poisson_3seed", "obj": "poisson", "seeds": 3, "hl": 0.0},
    {"name": "log_smear_w180_3seed", "obj": "log", "seeds": 3, "hl": 180.0},
    {"name": "poisson_w180_3seed", "obj": "poisson", "seeds": 3, "hl": 180.0},
]

FOLD_W = {"h1_last61": 0.65, "h1_prev61": 0.35}


def _build(X, names):
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
    return np.column_stack([X] + extra), ed


def _fit(Xr, yr, names, cfg, seed):
    Xf, ed = _build(Xr, names)
    y = np.clip(np.asarray(yr, dtype=float), 0.0, None)
    hl = float(cfg["hl"])
    if hl > 0.0:
        w = np.power(0.5, (float(np.max(ed)) - ed) / hl)
        w = np.clip(w, 1e-6, None)
    else:
        w = None
    models = []
    for k in range(int(cfg["seeds"])):
        rs = 1729 + 101 * k + int(abs(int(seed)) % 9000)
        if cfg["obj"] == "poisson":
            m = HistGradientBoostingRegressor(loss="poisson", random_state=rs,
                                              **BASE_KW)
            if w is None:
                m.fit(Xf, y)
            else:
                m.fit(Xf, y, sample_weight=w)
            models.append({"m": m, "log": False, "smear": 1.0})
        else:
            t = np.log1p(y)
            m = HistGradientBoostingRegressor(random_state=rs, **BASE_KW)
            if w is None:
                m.fit(Xf, t)
            else:
                m.fit(Xf, t, sample_weight=w)
            res = t - np.asarray(m.predict(Xf), dtype=float)
            ex = np.exp(np.clip(res, -20.0, 20.0))
            if w is None:
                s = float(np.mean(ex))
            else:
                s = float(np.sum(w * ex) / np.sum(w))
            if (not np.isfinite(s)) or s <= 0.0:
                s = 1.0
            models.append({"m": m, "log": True, "smear": s})
    return {"models": models, "cfg": cfg}


def _pred(mod, Xr, names):
    Xf, _ed = _build(Xr, names)
    acc = None
    for item in mod["models"]:
        p = np.asarray(item["m"].predict(Xf), dtype=float)
        if item["log"]:
            p = np.exp(np.clip(p, -20.0, 20.0)) * float(item["smear"]) - 1.0
        p = np.where(np.isfinite(p), p, 0.0)
        p = np.clip(p, 0.0, None)
        acc = p if acc is None else acc + p
    out = acc / float(len(mod["models"]))
    return np.clip(np.where(np.isfinite(out), out, 0.0), 0.0, None)


def fit_predict(train, inputs, seed):
    findings = {"candidates": [], "fallback_used": False, "notes": [],
                "transductive_use": False,
                "selection_rule": "weighted_rmse_0.65_last61_0.35_prev61_horizon_matched"}
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
                    mod = _fit(Xtr[tr_m], y[tr_m], names, cfg, seed)
                    p = _pred(mod, Xtr[va_m], names)
                    err = p - y[va_m]
                    rec["per_fold"][fname] = {
                        "rmse": float(np.sqrt(np.mean(err ** 2))),
                        "mae": float(np.mean(np.abs(err))),
                        "smear0": float(mod["models"][0]["smear"])}
                if len(rec["per_fold"]) == len(specs) and specs:
                    num = 0.0
                    den = 0.0
                    for fname, v in rec["per_fold"].items():
                        wf = float(FOLD_W.get(fname, 0.5))
                        num += wf * v["rmse"]
                        den += wf
                    rec["weighted_rmse"] = float(num / den)
                    rec["mean_rmse"] = float(np.mean([v["rmse"] for v in rec["per_fold"].values()]))
                    rec["mean_mae"] = float(np.mean([v["mae"] for v in rec["per_fold"].values()]))
                    scores[cfg["name"]] = rec["weighted_rmse"]
            except Exception as exc:
                rec["error"] = type(exc).__name__
                findings["notes"].append("candidate_failed_" + cfg["name"])
            findings["candidates"].append(rec)
        if scores:
            best_name = min(sorted(scores.keys()), key=lambda k: scores[k])
        else:
            best_name = "log_smear_1seed"
            findings["notes"].append("no_valid_candidate_score_used_log_smear_1seed_branch")
        best = [c for c in CANDS if c["name"] == best_name][0]
        findings["selected_branch"] = best_name
        findings["selected_weighted_rmse"] = scores.get(best_name)
        findings["parent_replicate_weighted_rmse"] = scores.get("log_smear_1seed")
        findings["poisson_weighted_rmse"] = scores.get("poisson_3seed")
        findings["n_scored_candidates"] = int(len(scores))
        final = _fit(Xtr, y, names, best, seed)
        findings["final_n_models"] = int(len(final["models"]))
        findings["final_smear_first"] = float(final["models"][0]["smear"])
        findings["final_half_life_days"] = float(best["hl"])
        findings["final_objective"] = best["obj"]
        Xf_chk, _e1 = _build(Xtr, names)
        Xf_ev, ed_ev = _build(Xev, ev_names)
        if Xf_ev.shape[1] != Xf_chk.shape[1]:
            raise ValueError("feature_width_mismatch")
        findings["train_ed_max"] = float(np.max(Xtr[:, ied]))
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
