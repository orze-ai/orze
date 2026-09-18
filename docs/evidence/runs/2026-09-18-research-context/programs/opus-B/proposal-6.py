import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

BASE_KW = {"max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 63,
           "l2_regularization": 1.0, "min_samples_leaf": 20}
FOLD_W = {"h1_last61": 0.65, "h1_prev61": 0.35}
SEEDBASE = 1729
SMOOTH_W = 10.0
FEATCFG = {"day_agg": True, "neigh": True}

CANDS = [
    {"name": "base_dayneigh", "anchor": None, "as_feat": False},
    {"name": "anchor_off_all", "anchor": "all", "as_feat": False},
    {"name": "anchor_off_recent90", "anchor": "recent90", "as_feat": False},
    {"name": "anchor_feat_all", "anchor": "all", "as_feat": True},
]

BLENDS = [
    {"name": "blend_base_anchor_all", "parts": ["base_dayneigh", "anchor_off_all"]},
    {"name": "blend_base_anchor_recent90",
     "parts": ["base_dayneigh", "anchor_off_recent90"]},
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


def _day_levels(X, y, ix):
    key = _daykey(X, ix)
    ll = np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None))
    lev = {}
    for d in np.unique(key):
        m = key == d
        lev[int(d)] = float(np.mean(ll[m]))
    return lev


def _fit_anchor(X, y, ix, mode):
    lev = _day_levels(X, y, ix)
    days = sorted(lev.keys())
    y1 = [d for d in days if d < 365]
    if len(y1) < 120:
        return None
    ad = np.asarray(y1, dtype=float)
    av = np.asarray([lev[int(d)] for d in y1], dtype=float)
    kk = []
    vv = []
    for doy in range(0, 366):
        w = np.abs(ad - float(doy)) <= SMOOTH_W
        if int(np.sum(w)) >= 3:
            kk.append(float(doy))
            vv.append(float(np.mean(av[w])))
    if len(kk) < 60:
        return None
    kka = np.asarray(kk, dtype=float)
    vva = np.asarray(vv, dtype=float)
    y2 = [d for d in days if d >= 365]
    g = 0.0
    ng = 0
    if y2:
        if mode == "recent90":
            cut = max(y2) - 90
            sel = [d for d in y2 if d >= cut]
        else:
            sel = y2
        res = []
        for d in sel:
            s = float(np.interp(float(d) - 365.0, kka, vva))
            res.append(lev[int(d)] - s)
        if res:
            g = float(np.median(np.asarray(res, dtype=float)))
            ng = len(res)
    if not np.isfinite(g):
        g = 0.0
    return {"kk": kka, "vv": vva, "g": float(g), "ng": int(ng), "mode": str(mode)}


def _anchor_raw(anch, X, ix):
    ed = _daykey(X, ix).astype(float)
    doy = np.where(ed >= 365.0, ed - 365.0, ed)
    base = np.interp(doy, anch["kk"], anch["vv"])
    out = base + np.where(ed >= 365.0, float(anch["g"]), 0.0)
    return np.asarray(out, dtype=float)


def _fit_one(Xr, yr, names, cfg, seed):
    X = np.asarray(Xr, dtype=float)
    ix = _ixmap(names)
    y = np.clip(np.asarray(yr, dtype=float), 0.0, None)
    t = np.log1p(y)
    anch = None
    if cfg.get("anchor"):
        anch = _fit_anchor(X, y, ix, cfg["anchor"])
    off = np.zeros(X.shape[0], dtype=float)
    feat = None
    c = 0.0
    if anch is not None:
        raw = _anchor_raw(anch, X, ix)
        c = float(np.mean(raw))
        if cfg.get("as_feat"):
            feat = raw
        else:
            off = raw - c
    Xf = _build(X, names, FEATCFG)
    if feat is not None:
        Xf = np.column_stack([Xf, feat])
    tt = t - off
    m = HistGradientBoostingRegressor(random_state=_rs(seed, 0), **BASE_KW)
    m.fit(Xf, tt)
    res = tt - np.asarray(m.predict(Xf), dtype=float)
    s = float(np.mean(np.exp(np.clip(res, -20.0, 20.0))))
    if (not np.isfinite(s)) or s <= 0.0:
        s = 1.0
    return {"m": m, "smear": s, "anch": anch, "c": c, "cfg": cfg,
            "anchor_requested": bool(cfg.get("anchor")),
            "anchor_ok": bool(anch is not None)}


def _pred_one(mod, Xr, names):
    X = np.asarray(Xr, dtype=float)
    ix = _ixmap(names)
    off = np.zeros(X.shape[0], dtype=float)
    feat = None
    if mod["anch"] is not None:
        raw = _anchor_raw(mod["anch"], X, ix)
        if mod["cfg"].get("as_feat"):
            feat = raw
        else:
            off = raw - float(mod["c"])
    Xf = _build(X, names, FEATCFG)
    if feat is not None:
        Xf = np.column_stack([Xf, feat])
    p = np.asarray(mod["m"].predict(Xf), dtype=float) + off
    p = np.exp(np.clip(p, -20.0, 20.0)) * float(mod["smear"]) - 1.0
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, None)


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
        per[fname] = {"rmse": r, "mae": float(np.mean(np.abs(err))),
                      "bias": float(np.mean(err))}
        w = float(FOLD_W.get(fname, 0.5))
        num += w * r
        den += w
    if den <= 0.0:
        return None, per
    return float(num / den), per


def fit_predict(train, inputs, seed):
    findings = {"candidates": [], "blends": [], "notes": [], "fallback_used": False,
                "selection_rule": "weighted_rmse_0.65_last61_0.35_prev61_horizon_matched",
                "anchor_note": "anchor_uses_training_labels_only_plus_supplied_evaluation_calendar",
                "transductive_note": "day_agg_and_neigh_use_unlabeled_covariates_grouped_by_date_including_evaluation_rows_no_labels"}
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
        findings["n_train_dates"] = int(nd)
        specs = []
        if nd >= 150:
            specs.append(("h1_last61", order[:nd - 61], order[nd - 61:]))
        if nd >= 250:
            specs.append(("h1_prev61", order[:nd - 122], order[nd - 122:nd - 61]))
        findings["folds"] = [s[0] for s in specs]
        if not specs:
            findings["notes"].append("too_few_dates_for_time_blocked_validation")
        try:
            ix_full = _ixmap(names)
            a_all = _fit_anchor(Xtr, y, ix_full, "all")
            a_rec = _fit_anchor(Xtr, y, ix_full, "recent90")
            findings["anchor_full_train"] = {
                "all_g": None if a_all is None else float(a_all["g"]),
                "all_n_days_2012": None if a_all is None else int(a_all["ng"]),
                "recent90_g": None if a_rec is None else float(a_rec["g"]),
                "recent90_n_days_2012": None if a_rec is None else int(a_rec["ng"])}
        except Exception as exca:
            findings["notes"].append("anchor_diagnostic_failed_" + type(exca).__name__)
        preds = {}
        for cfg in CANDS:
            rec = {"name": cfg["name"], "per_fold": {}, "anchor_ok_per_fold": {}}
            try:
                for fname, trd, vad in specs:
                    tr_m = np.isin(gg, np.asarray(trd))
                    va_m = np.isin(gg, np.asarray(vad))
                    mod = _fit_one(Xtr[tr_m], y[tr_m], names, cfg, seed)
                    rec["anchor_ok_per_fold"][fname] = bool(mod["anchor_ok"])
                    if mod["anchor_requested"] and not mod["anchor_ok"]:
                        findings["notes"].append(
                            "anchor_unavailable_fell_back_to_base_" + cfg["name"] + "_" + fname)
                    p = _pred_one(mod, Xtr[va_m], names)
                    preds[(cfg["name"], fname)] = p
                    err = p - y[va_m]
                    rec["per_fold"][fname] = {
                        "rmse": float(np.sqrt(np.mean(err ** 2))),
                        "mae": float(np.mean(np.abs(err))),
                        "bias": float(np.mean(err))}
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
            best_name = "base_dayneigh"
            parts = ["base_dayneigh"]
            findings["selected_weighted_rmse"] = None
            findings["notes"].append("no_scored_option_used_base_dayneigh_branch")
        findings["selected_branch"] = str(best_name)
        findings["selected_parts"] = [str(p) for p in parts]
        findings["n_scored_options"] = int(len(options))
        for nm in ("base_dayneigh", "anchor_off_all", "anchor_off_recent90",
                   "anchor_feat_all"):
            if nm in options:
                findings[nm + "_weighted_rmse"] = float(options[nm][0])
        acc = None
        used_anchor = False
        for nm in parts:
            cfg = [c for c in CANDS if c["name"] == nm][0]
            mod = _fit_one(Xtr, y, names, cfg, seed)
            if mod["anchor_requested"] and not mod["anchor_ok"]:
                findings["notes"].append("final_anchor_unavailable_fell_back_to_base_" + nm)
            if mod["anchor_ok"]:
                used_anchor = True
                findings["final_anchor_g_" + nm] = float(mod["anch"]["g"])
            p = np.asarray(_pred_one(mod, Xev, ev_names), dtype=float)
            if p.shape[0] != Xev.shape[0]:
                raise ValueError("prediction_length_mismatch")
            acc = p if acc is None else acc + p
        pred = acc / float(len(parts))
        pred = np.clip(np.where(np.isfinite(pred), pred, 0.0), 0.0, None)
        findings["final_used_anchor"] = bool(used_anchor)
        findings["transductive_use"] = True
        findings["train_ed_max"] = float(np.max(Xtr[:, ied]))
        iev = ev_names.index("elapsed_day")
        findings["eval_ed_min"] = float(np.min(Xev[:, iev]))
        findings["eval_ed_max"] = float(np.max(Xev[:, iev]))
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
