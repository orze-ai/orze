import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

BASE_KW = {"max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 63,
           "l2_regularization": 1.0, "min_samples_leaf": 20}
DAY_KW = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15,
          "l2_regularization": 1.0, "min_samples_leaf": 10}
SHARE_KW = {"max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31,
            "l2_regularization": 1.0, "min_samples_leaf": 20}
FOLD_W = {"h1_last61": 0.65, "h1_prev61": 0.35}
SEEDBASE = 1729

CANDS = [
    {"name": "flat_rep", "day_agg": False, "neigh": False, "hier": False},
    {"name": "flat_day", "day_agg": True, "neigh": False, "hier": False},
    {"name": "flat_day_neigh", "day_agg": True, "neigh": True, "hier": False},
    {"name": "hier_day_share", "day_agg": True, "neigh": False, "hier": True},
]

BLENDS = [
    {"name": "blend_rep_day", "parts": ["flat_rep", "flat_day"]},
    {"name": "blend_day_hier", "parts": ["flat_day", "hier_day_share"]},
    {"name": "blend_rep_dayneigh_hier",
     "parts": ["flat_rep", "flat_day_neigh", "hier_day_share"]},
]


def _ixmap(names):
    d = {}
    for i, n in enumerate(names):
        d[n] = i
    return d


def _core(X, ix):
    hr = X[:, ix["hr"]]
    ed = X[:, ix["elapsed_day"]]
    wk = X[:, ix["workingday"]]
    doy = np.mod(ed, 365.25)
    return [np.sin(2.0 * np.pi * hr / 24.0), np.cos(2.0 * np.pi * hr / 24.0),
            np.sin(4.0 * np.pi * hr / 24.0), np.cos(4.0 * np.pi * hr / 24.0),
            np.sin(2.0 * np.pi * doy / 365.25), np.cos(2.0 * np.pi * doy / 365.25),
            hr + 24.0 * wk,
            X[:, ix["temp"]] * X[:, ix["hum"]],
            X[:, ix["atemp"]] - X[:, ix["temp"]]]


def _daykey(X, ix):
    return np.round(X[:, ix["elapsed_day"]]).astype(np.int64)


def _day_ctx(X, ix, key):
    n = X.shape[0]
    out = np.full((n, 7), np.nan)
    temp = X[:, ix["temp"]]
    hum = X[:, ix["hum"]]
    wind = X[:, ix["windspeed"]]
    ws = X[:, ix["weathersit"]]
    for d in np.unique(key):
        m = key == d
        out[m, 0] = float(np.mean(temp[m]))
        out[m, 1] = float(np.max(temp[m]))
        out[m, 2] = float(np.mean(hum[m]))
        out[m, 3] = float(np.max(hum[m]))
        out[m, 4] = float(np.mean(wind[m]))
        out[m, 5] = float(np.max(ws[m]))
        out[m, 6] = float(np.mean(ws[m] >= 3.0))
    return out


def _neigh(X, ix, key):
    n = X.shape[0]
    hr = np.round(X[:, ix["hr"]]).astype(np.int64)
    pos = {}
    for i in range(n):
        pos[(int(key[i]), int(hr[i]))] = i
    out = np.full((n, 4), np.nan)
    ws = X[:, ix["weathersit"]]
    temp = X[:, ix["temp"]]
    hum = X[:, ix["hum"]]
    for i in range(n):
        p = pos.get((int(key[i]), int(hr[i]) - 1))
        q = pos.get((int(key[i]), int(hr[i]) + 1))
        if p is not None:
            out[i, 0] = ws[p]
            out[i, 1] = temp[p]
            out[i, 2] = hum[p]
        if q is not None:
            out[i, 3] = ws[q]
    return out


def _build(Xin, names, cfg):
    X = np.asarray(Xin, dtype=float)
    ix = _ixmap(names)
    key = _daykey(X, ix)
    parts = [X] + _core(X, ix)
    if cfg.get("day_agg"):
        parts.append(_day_ctx(X, ix, key))
    if cfg.get("neigh"):
        parts.append(_neigh(X, ix, key))
    return np.column_stack(parts)


def _rs(seed, off):
    return SEEDBASE + off + int(abs(int(seed)) % 9000)


def _fit_flat(Xr, yr, names, cfg, seed):
    Xf = _build(Xr, names, cfg)
    y = np.clip(np.asarray(yr, dtype=float), 0.0, None)
    t = np.log1p(y)
    m = HistGradientBoostingRegressor(random_state=_rs(seed, 0), **BASE_KW)
    m.fit(Xf, t)
    res = t - np.asarray(m.predict(Xf), dtype=float)
    s = float(np.mean(np.exp(np.clip(res, -20.0, 20.0))))
    if (not np.isfinite(s)) or s <= 0.0:
        s = 1.0
    return {"kind": "flat", "m": m, "smear": s, "cfg": cfg}


def _pred_flat(mod, Xr, names):
    Xf = _build(Xr, names, mod["cfg"])
    p = np.asarray(mod["m"].predict(Xf), dtype=float)
    p = np.exp(np.clip(p, -20.0, 20.0)) * float(mod["smear"]) - 1.0
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, None)


def _day_table(X, ix, key):
    uq = np.unique(key)
    rows = []
    for d in uq:
        m = key == d
        sub = X[m]
        dd = float(d)
        rows.append([dd,
                     float(np.mean(sub[:, ix["temp"]])),
                     float(np.max(sub[:, ix["temp"]])),
                     float(np.mean(sub[:, ix["hum"]])),
                     float(np.max(sub[:, ix["hum"]])),
                     float(np.mean(sub[:, ix["windspeed"]])),
                     float(np.max(sub[:, ix["weathersit"]])),
                     float(np.mean(sub[:, ix["weathersit"]] >= 3.0)),
                     float(sub[0, ix["season"]]),
                     float(sub[0, ix["yr"]]),
                     float(sub[0, ix["mnth"]]),
                     float(sub[0, ix["holiday"]]),
                     float(sub[0, ix["weekday"]]),
                     float(sub[0, ix["workingday"]]),
                     float(np.sin(2.0 * np.pi * (dd % 365.25) / 365.25)),
                     float(np.cos(2.0 * np.pi * (dd % 365.25) / 365.25)),
                     float(sub.shape[0])])
    return uq, np.asarray(rows, dtype=float)


def _fit_hier(Xr, yr, names, cfg, seed):
    X = np.asarray(Xr, dtype=float)
    ix = _ixmap(names)
    y = np.clip(np.asarray(yr, dtype=float), 0.0, None)
    key = _daykey(X, ix)
    uq, dtab = _day_table(X, ix, key)
    kpos = {}
    for j, d in enumerate(uq):
        kpos[int(d)] = j
    tot = np.zeros(uq.shape[0], dtype=float)
    for i in range(X.shape[0]):
        tot[kpos[int(key[i])]] += y[i]
    md = HistGradientBoostingRegressor(random_state=_rs(seed, 11), **DAY_KW)
    md.fit(dtab, np.log1p(tot))
    denom = np.asarray([max(tot[kpos[int(k)]], 1.0) for k in key], dtype=float)
    share = np.clip(y / denom, 1e-6, None)
    Xf = _build(X, names, {"day_agg": True})
    ms = HistGradientBoostingRegressor(random_state=_rs(seed, 23), **SHARE_KW)
    ms.fit(Xf, np.log(share))
    return {"kind": "hier", "md": md, "ms": ms}


def _pred_hier(mod, Xr, names):
    X = np.asarray(Xr, dtype=float)
    ix = _ixmap(names)
    key = _daykey(X, ix)
    uq, dtab = _day_table(X, ix, key)
    dp = np.asarray(mod["md"].predict(dtab), dtype=float)
    dp = np.expm1(np.clip(dp, -20.0, 20.0))
    dp = np.clip(np.where(np.isfinite(dp), dp, 0.0), 0.0, None)
    Xf = _build(X, names, {"day_agg": True})
    sp = np.asarray(mod["ms"].predict(Xf), dtype=float)
    sp = np.exp(np.clip(sp, -20.0, 5.0))
    sp = np.where(np.isfinite(sp), sp, 1e-6)
    out = np.zeros(X.shape[0], dtype=float)
    for j, d in enumerate(uq):
        m = key == int(d)
        ssum = float(np.sum(sp[m]))
        cnt = int(np.sum(m))
        if ssum <= 0.0 or cnt <= 0:
            out[m] = dp[j] / float(max(cnt, 1))
        else:
            out[m] = dp[j] * sp[m] / ssum
    return np.clip(np.where(np.isfinite(out), out, 0.0), 0.0, None)


def _fit_one(Xr, yr, names, cfg, seed):
    if cfg.get("hier"):
        return _fit_hier(Xr, yr, names, cfg, seed)
    return _fit_flat(Xr, yr, names, cfg, seed)


def _pred_one(mod, Xr, names):
    if mod["kind"] == "hier":
        return _pred_hier(mod, Xr, names)
    return _pred_flat(mod, Xr, names)


def _wscore(preds, specs, gg, y, parts):
    num = 0.0
    den = 0.0
    per = {}
    for fname, trd, vad in specs:
        va_m = np.isin(gg, np.asarray(vad))
        acc = None
        for nm in parts:
            pp = preds.get((nm, fname))
            if pp is None:
                return None, {}
            pp = np.asarray(pp, dtype=float)
            acc = pp if acc is None else acc + pp
        p = acc / float(len(parts))
        err = p - y[va_m]
        r = float(np.sqrt(np.mean(err ** 2)))
        per[fname] = {"rmse": r, "mae": float(np.mean(np.abs(err)))}
        w = float(FOLD_W.get(fname, 0.5))
        num += w * r
        den += w
    if den <= 0.0:
        return None, per
    return float(num / den), per


def fit_predict(train, inputs, seed):
    findings = {"candidates": [], "blends": [], "notes": [], "fallback_used": False,
                "selection_rule": "weighted_rmse_0.65_last61_0.35_prev61_horizon_matched",
                "transductive_note": "day_agg_neigh_hier_use_unlabeled_covariates_grouped_by_date_including_evaluation_rows_no_labels"}
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
        uq = np.unique(gg).tolist()
        keyd = {}
        for d in uq:
            keyd[d] = float(np.mean(Xtr[gg == d, ied]))
        order = sorted(uq, key=lambda d: keyd[d])
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
        preds = {}
        for cfg in CANDS:
            rec = {"name": cfg["name"], "per_fold": {}}
            try:
                for fname, trd, vad in specs:
                    tr_m = np.isin(gg, np.asarray(trd))
                    va_m = np.isin(gg, np.asarray(vad))
                    mod = _fit_one(Xtr[tr_m], y[tr_m], names, cfg, seed)
                    p = _pred_one(mod, Xtr[va_m], names)
                    preds[(cfg["name"], fname)] = p
                    err = p - y[va_m]
                    rec["per_fold"][fname] = {
                        "rmse": float(np.sqrt(np.mean(err ** 2))),
                        "mae": float(np.mean(np.abs(err)))}
            except Exception as exc:
                rec["error"] = type(exc).__name__
                findings["notes"].append("candidate_failed_" + cfg["name"])
            findings["candidates"].append(rec)
        options = {}
        for cfg in CANDS:
            s, per = _wscore(preds, specs, gg, y, [cfg["name"]])
            if s is not None:
                options[cfg["name"]] = (s, [cfg["name"]])
                for rec in findings["candidates"]:
                    if rec["name"] == cfg["name"]:
                        rec["weighted_rmse"] = s
        for b in BLENDS:
            s, per = _wscore(preds, specs, gg, y, b["parts"])
            if s is not None:
                options[b["name"]] = (s, list(b["parts"]))
                findings["blends"].append({"name": b["name"], "weighted_rmse": s,
                                           "per_fold": per})
        if options:
            best_name = min(sorted(options.keys()), key=lambda k: options[k][0])
            parts = list(options[best_name][1])
            findings["selected_weighted_rmse"] = float(options[best_name][0])
        else:
            best_name = "flat_rep"
            parts = ["flat_rep"]
            findings["selected_weighted_rmse"] = None
            findings["notes"].append("no_scored_option_used_flat_rep_branch")
        findings["selected_branch"] = best_name
        findings["selected_parts"] = parts
        findings["n_scored_options"] = int(len(options))
        if "flat_rep" in options:
            findings["parent_replicate_weighted_rmse"] = float(options["flat_rep"][0])
        if "flat_day" in options:
            findings["flat_day_weighted_rmse"] = float(options["flat_day"][0])
        if "flat_day_neigh" in options:
            findings["flat_day_neigh_weighted_rmse"] = float(options["flat_day_neigh"][0])
        if "hier_day_share" in options:
            findings["hier_weighted_rmse"] = float(options["hier_day_share"][0])
        trans = False
        acc = None
        for nm in parts:
            cfg = [c for c in CANDS if c["name"] == nm][0]
            if cfg.get("day_agg") or cfg.get("neigh") or cfg.get("hier"):
                trans = True
            mod = _fit_one(Xtr, y, names, cfg, seed)
            p = np.asarray(_pred_one(mod, Xev, ev_names), dtype=float)
            if p.shape[0] != Xev.shape[0]:
                raise ValueError("prediction_length_mismatch")
            acc = p if acc is None else acc + p
        pred = acc / float(len(parts))
        pred = np.clip(np.where(np.isfinite(pred), pred, 0.0), 0.0, None)
        findings["transductive_use"] = bool(trans)
        findings["train_ed_max"] = float(np.max(Xtr[:, ied]))
        findings["eval_ed_min"] = float(np.min(Xev[:, ev_names.index("elapsed_day")]))
        findings["eval_ed_max"] = float(np.max(Xev[:, ev_names.index("elapsed_day")]))
        findings["pred_rows"] = int(pred.shape[0])
        findings["pred_mean"] = float(np.mean(pred))
        findings["pred_max"] = float(np.max(pred))
        findings["train_y_mean"] = float(np.mean(y))
        return {"prediction": pred.tolist(), "findings": findings}
    except Exception as exc:
        findings["fallback_used"] = True
        findings["fallback_reason"] = type(exc).__name__
        findings["notes"].append("broad_handler_reference_hgb_250_0.08_31_on_raw_columns_not_intended_component")
        fb = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                           max_leaf_nodes=31,
                                           l2_regularization=1,
                                           random_state=1729)
        fb.fit(Xtr, y)
        p = np.asarray(fb.predict(Xev), dtype=float)
        p = np.clip(np.where(np.isfinite(p), p, float(np.mean(y))), 0.0, None)
        return {"prediction": p.tolist(), "findings": findings}
