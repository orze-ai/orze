import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

CANDS = [
    {"name": "ref_raw_plain", "log": False, "eng": False, "drop_ed": False,
     "iters": 250, "lr": 0.08, "leaves": 31, "smear": False},
    {"name": "raw_eng", "log": False, "eng": True, "drop_ed": False,
     "iters": 400, "lr": 0.06, "leaves": 63, "smear": False},
    {"name": "log_plain", "log": True, "eng": False, "drop_ed": False,
     "iters": 400, "lr": 0.06, "leaves": 31, "smear": False},
    {"name": "log_eng", "log": True, "eng": True, "drop_ed": False,
     "iters": 400, "lr": 0.06, "leaves": 63, "smear": False},
    {"name": "log_eng_smear", "log": True, "eng": True, "drop_ed": False,
     "iters": 400, "lr": 0.06, "leaves": 63, "smear": True},
    {"name": "log_eng_noed", "log": True, "eng": True, "drop_ed": True,
     "iters": 400, "lr": 0.06, "leaves": 63, "smear": False},
]


def _features(X, names, cfg):
    X = np.asarray(X, dtype=float)
    ix = {}
    for i, n in enumerate(names):
        ix[n] = i
    keep = [i for i, n in enumerate(names) if n != "elapsed_day"]
    if not cfg["eng"]:
        if cfg["drop_ed"] and "elapsed_day" in ix:
            return X[:, keep]
        return X
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
    base = X[:, keep] if (cfg["drop_ed"] and "elapsed_day" in ix) else X
    return np.column_stack([base] + extra)


def _fit_model(Xtr, ytr, cfg, seed):
    rs = int(abs(int(seed)) % 90000) + 1729
    m = HistGradientBoostingRegressor(max_iter=int(cfg["iters"]),
                                      learning_rate=float(cfg["lr"]),
                                      max_leaf_nodes=int(cfg["leaves"]),
                                      l2_regularization=1.0,
                                      min_samples_leaf=20,
                                      random_state=rs)
    t = np.log1p(np.clip(ytr, 0.0, None)) if cfg["log"] else ytr
    m.fit(Xtr, t)
    smear = 1.0
    if cfg["log"] and cfg["smear"]:
        res = t - m.predict(Xtr)
        s = float(np.mean(np.exp(np.clip(res, -20.0, 20.0))))
        if np.isfinite(s) and s > 0.0:
            smear = s
    return m, smear


def _predict(m, smear, X, cfg):
    p = np.asarray(m.predict(X), dtype=float)
    if cfg["log"]:
        p = np.exp(np.clip(p, -20.0, 20.0)) * smear - 1.0
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, None)


def fit_predict(train, inputs, seed):
    findings = {"candidates": [], "fallback_used": False, "notes": []}
    Xtr_raw = np.asarray(train["X"], dtype=float)
    y = np.asarray(train["y"], dtype=float)
    names = list(train["feature_names"])
    Xev_raw = np.asarray(inputs["X"], dtype=float)
    ev_names = list(inputs["feature_names"])
    try:
        if ev_names != names:
            findings["notes"].append("feature_name_mismatch_used_own_order")
        groups = np.asarray(train["groups"])
        uniq = np.unique(groups)
        nd = int(uniq.shape[0])
        folds = []
        if nd >= 150:
            folds.append((set(uniq[:nd - 61].tolist()), set(uniq[nd - 61:].tolist())))
        if nd >= 240:
            folds.append((set(uniq[:nd - 122].tolist()),
                          set(uniq[nd - 122:nd - 61].tolist())))
        findings["n_train_dates"] = nd
        findings["n_folds"] = len(folds)
        scores = {}
        if folds:
            glist = groups.tolist()
            for cfg in CANDS:
                rec = {"name": cfg["name"], "fold_rmse": [], "fold_mae": []}
                try:
                    Xf = _features(Xtr_raw, names, cfg)
                    for tr_d, va_d in folds:
                        tr_m = np.array([g in tr_d for g in glist])
                        va_m = np.array([g in va_d for g in glist])
                        m, sm = _fit_model(Xf[tr_m], y[tr_m], cfg, seed)
                        p = _predict(m, sm, Xf[va_m], cfg)
                        err = p - y[va_m]
                        rec["fold_rmse"].append(float(np.sqrt(np.mean(err ** 2))))
                        rec["fold_mae"].append(float(np.mean(np.abs(err))))
                    rec["mean_rmse"] = float(np.mean(rec["fold_rmse"]))
                    rec["mean_mae"] = float(np.mean(rec["fold_mae"]))
                    scores[cfg["name"]] = rec["mean_rmse"]
                except Exception as exc:
                    rec["error"] = type(exc).__name__
                findings["candidates"].append(rec)
        else:
            findings["notes"].append("too_few_dates_for_time_blocked_validation")
        if scores:
            best_name = min(sorted(scores.keys()), key=lambda k: scores[k])
        else:
            best_name = "ref_raw_plain"
            findings["notes"].append("no_valid_candidate_score_used_reference_branch")
        best = [c for c in CANDS if c["name"] == best_name][0]
        findings["selected_branch"] = best_name
        findings["selected_internal_rmse"] = scores.get(best_name)
        findings["reference_internal_rmse"] = scores.get("ref_raw_plain")
        Xf_full = _features(Xtr_raw, names, best)
        Xf_ev = _features(Xev_raw, ev_names, best)
        if Xf_ev.shape[1] != Xf_full.shape[1]:
            raise ValueError("feature_width_mismatch")
        model, smear = _fit_model(Xf_full, y, best, seed)
        findings["smearing_factor"] = float(smear)
        pred = _predict(model, smear, Xf_ev, best)
        findings["pred_mean"] = float(np.mean(pred))
        findings["pred_max"] = float(np.max(pred))
        findings["pred_rows"] = int(pred.shape[0])
        findings["train_y_mean"] = float(np.mean(y))
        return {"prediction": pred.tolist(), "findings": findings}
    except Exception as exc:
        findings["fallback_used"] = True
        findings["fallback_reason"] = type(exc).__name__
        findings["notes"].append("broad_handler_reference_hgb_on_raw_columns")
        fb = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                           max_leaf_nodes=31,
                                           l2_regularization=1,
                                           random_state=1729)
        fb.fit(Xtr_raw, y)
        p = np.asarray(fb.predict(Xev_raw), dtype=float)
        p = np.clip(np.where(np.isfinite(p), p, float(np.mean(y))), 0.0, None)
        return {"prediction": p.tolist(), "findings": findings}
