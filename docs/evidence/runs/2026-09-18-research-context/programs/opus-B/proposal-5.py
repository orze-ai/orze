import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

BASE_KW = {"max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 63,
           "l2_regularization": 1.0, "min_samples_leaf": 20}
FOLD_W = {"h1_last61": 0.65, "h1_prev61": 0.35}

CANDS = [
    {"name": "flat_rep", "day_agg": False, "neigh": False, "log": True, "hl": None},
    {"name": "flat_day", "day_agg": True, "neigh": False, "log": True, "hl": None},
    {"name": "flat_day_neigh", "day_agg": True, "neigh": True, "log": True, "hl": None},
    {"name": "dayneigh_rec120", "day_agg": True, "neigh": True, "log": True, "hl": 120.0},
    {"name": "dayneigh_rec60", "day_agg": True, "neigh": True, "log": True, "hl": 60.0},
    {"name": "dayneigh_raw", "day_agg": True, "neigh": True, "log": False, "hl": None},
]
FOLD_CANDS = ["flat_rep", "flat_day", "flat_day_neigh"]


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


def _build(X, names, cfg):
    ix = _ixmap(names)
    key = _daykey(X, ix)
    parts = [X] + _core(X, ix)
    if cfg.get("day_agg"):
        parts.append(_day_ctx(X, ix, key))
    if cfg.get("neigh"):
        parts.append(_neigh(X, ix, key))
    return np.column_stack(parts)


def _weights(X, names, hl):
    if not hl:
        return None
    ed = X[:, _ixmap(names)["elapsed_day"]]
    w = np.exp(-(float(np.max(ed)) - ed) / float(hl))
    return np.clip(w, 1e-4, None)


def _fit(X, y, names, cfg, seed):
    Xf = _build(X, names, cfg)
    yy = np.clip(np.asarray(y, dtype=float), 0.0, None)
    use_log = bool(cfg.get("log", True))
    t = np.log1p(yy) if use_log else yy
    m = HistGradientBoostingRegressor(random_state=1729 + int(abs(int(seed)) % 9000),
                                      **BASE_KW)
    m.fit(Xf, t, sample_weight=_weights(X, names, cfg.get("hl")))
    smear = 1.0
    if use_log:
        res = t - np.asarray(m.predict(Xf), dtype=float)
        s = float(np.mean(np.exp(np.clip(res, -20.0, 20.0))))
        if np.isfinite(s) and s > 0.0:
            smear = s
    return {"m": m, "cfg": cfg, "smear": smear, "log": use_log}


def _pred(mod, X, names):
    Xf = _build(X, names, mod["cfg"])
    p = np.asarray(mod["m"].predict(Xf), dtype=float)
    if mod["log"]:
        p = np.exp(np.clip(p, -20.0, 20.0)) * float(mod["smear"]) - 1.0
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, None)


def _metrics(p, y, g):
    e = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    res = {"rmse": float(np.sqrt(np.mean(e ** 2))),
           "mae": float(np.mean(np.abs(e))),
           "bias": float(np.mean(e))}
    if g is not None:
        vals = []
        for d in np.unique(g):
            m = g == d
            vals.append(float(np.mean(np.abs(e[m]))))
        res["group_mae"] = float(np.mean(vals))
    return res


def _date_order(X, names, g):
    ied = _ixmap(names)["elapsed_day"]
    uq = np.unique(g).tolist()
    keyd = {}
    for d in uq:
        keyd[d] = float(np.mean(X[g == d, ied]))
    return sorted(uq, key=lambda d: keyd[d]), keyd


def analyze(data, history, seed):
    out = {"question": "fold_ranking_vs_actual_development_ranking_and_recency_or_level_correction",
           "fallback_used": False, "notes": [],
           "label_use_note": "development_labels_are_supplied_to_analyze_confirmation_labels_never_used",
           "transductive_note": "day_aggregate_and_neighbour_hour_features_use_unlabeled_covariates_grouped_by_date"}
    try:
        try:
            out["n_history_records"] = int(len(history))
        except Exception as exc0:
            out["notes"].append("history_len_unavailable_" + type(exc0).__name__)
        tr = data["train"]
        dv = data["development"]
        Xtr = np.asarray(tr["X"], dtype=float)
        ytr = np.asarray(tr["y"], dtype=float)
        names = list(tr["feature_names"])
        gtr = np.asarray([str(v) for v in tr["groups"]])
        Xd = np.asarray(dv["X"], dtype=float)
        yd = np.asarray(dv["y"], dtype=float)
        dnames = list(dv["feature_names"])
        gd = np.asarray([str(v) for v in dv["groups"]])
        if dnames != names:
            out["notes"].append("dev_feature_order_differs_used_each_own_index")
        out["n_train_rows"] = int(Xtr.shape[0])
        out["n_dev_rows"] = int(Xd.shape[0])
        out["train_y_mean"] = float(np.mean(ytr))
        out["dev_y_mean"] = float(np.mean(yd))
        order, keyd = _date_order(Xtr, names, gtr)
        nd = len(order)
        out["n_train_dates"] = int(nd)
        out["n_dev_dates"] = int(np.unique(gd).shape[0])
        specs = []
        if nd >= 150:
            specs.append(("h1_last61", order[:nd - 61], order[nd - 61:]))
        if nd >= 250:
            specs.append(("h1_prev61", order[:nd - 122], order[nd - 122:nd - 61]))
        out["folds"] = [s[0] for s in specs]
        dev_res = {}
        fold_res = {}
        keep_pred = {}
        for cfg in CANDS:
            nm = cfg["name"]
            try:
                mod = _fit(Xtr, ytr, names, cfg, seed)
                p = _pred(mod, Xd, dnames)
                dev_res[nm] = _metrics(p, yd, gd)
                if nm in ("flat_day_neigh", "flat_day"):
                    keep_pred[nm] = p
            except Exception as exc:
                out["notes"].append("dev_fit_failed_" + nm + "_" + type(exc).__name__)
            if nm in FOLD_CANDS and specs:
                num = 0.0
                den = 0.0
                per = {}
                ok = True
                for fname, trd, vad in specs:
                    try:
                        tm = np.isin(gtr, np.asarray(trd))
                        vm = np.isin(gtr, np.asarray(vad))
                        fmod = _fit(Xtr[tm], ytr[tm], names, cfg, seed)
                        fp = _pred(fmod, Xtr[vm], names)
                        per[fname] = _metrics(fp, ytr[vm], gtr[vm])
                        w = float(FOLD_W.get(fname, 0.5))
                        num += w * per[fname]["rmse"]
                        den += w
                    except Exception as exc2:
                        ok = False
                        out["notes"].append("fold_fit_failed_" + nm + "_" + type(exc2).__name__)
                if ok and den > 0.0:
                    fold_res[nm] = {"weighted_rmse": float(num / den), "per_fold": per}
        out["dev_metrics"] = dev_res
        out["fold_metrics"] = fold_res
        if dev_res:
            dev_rank = sorted(dev_res.keys(), key=lambda k: dev_res[k]["rmse"])
            out["dev_rank_by_rmse"] = dev_rank
            out["dev_best"] = dev_rank[0]
        if fold_res:
            fr = sorted(fold_res.keys(), key=lambda k: fold_res[k]["weighted_rmse"])
            out["fold_rank_by_weighted_rmse"] = fr
            out["fold_best_among_fold_cands"] = fr[0]
            sub = [k for k in out.get("dev_rank_by_rmse", []) if k in fold_res]
            out["dev_rank_restricted_to_fold_cands"] = sub
            out["fold_and_dev_argmin_agree"] = bool(sub and sub[0] == fr[0])
            if "flat_day_neigh" in fold_res and "flat_day_neigh" in dev_res:
                out["fold_over_dev_rmse_ratio_flat_day_neigh"] = float(
                    fold_res["flat_day_neigh"]["weighted_rmse"]
                    / max(dev_res["flat_day_neigh"]["rmse"], 1e-9))
        base = keep_pred.get("flat_day_neigh")
        if base is not None:
            ixd = _ixmap(dnames)
            hrd = np.round(Xd[:, ixd["hr"]]).astype(np.int64)
            e = base - yd
            buck = {}
            for lab, lo, hi in (("h00_06", 0, 6), ("h07_10", 7, 10),
                               ("h11_16", 11, 16), ("h17_23", 17, 23)):
                m = (hrd >= lo) & (hrd <= hi)
                if int(np.sum(m)) > 0:
                    buck[lab] = {"bias": float(np.mean(e[m])),
                                 "rmse": float(np.sqrt(np.mean(e[m] ** 2))),
                                 "n": int(np.sum(m))}
            out["dev_bias_by_hour_bucket"] = buck
            dord, _dk = _date_order(Xd, dnames, gd)
            half = len(dord) // 2
            first = np.isin(gd, np.asarray(dord[:half]))
            second = np.isin(gd, np.asarray(dord[half:]))
            out["dev_bias_first_half_dates"] = float(np.mean(e[first]))
            out["dev_bias_second_half_dates"] = float(np.mean(e[second]))
            ratios = []
            for d in dord:
                m = gd == d
                sy = float(np.sum(yd[m]))
                if sy > 0.0:
                    ratios.append(float(np.sum(base[m])) / sy)
            if ratios:
                out["dev_daily_total_pred_over_actual_mean"] = float(np.mean(ratios))
                out["dev_daily_total_pred_over_actual_median"] = float(np.median(ratios))
            try:
                tail = order[nd - 28:]
                head = order[:nd - 28]
                tm = np.isin(gtr, np.asarray(head))
                vm = np.isin(gtr, np.asarray(tail))
                cmod = _fit(Xtr[tm], ytr[tm], names, CANDS[2], seed)
                cp = _pred(cmod, Xtr[vm], names)
                sp = float(np.sum(cp))
                fac = float(np.sum(ytr[vm])) / sp if sp > 0.0 else 1.0
                out["tail28_level_factor"] = fac
                out["tail28_uncalibrated"] = _metrics(cp, ytr[vm], gtr[vm])
                cal = np.clip(base * fac, 0.0, None)
                out["dev_metrics_level_calibrated"] = _metrics(cal, yd, gd)
                rb = dev_res.get("flat_day_neigh", {}).get("rmse")
                if rb:
                    out["level_calibration_rel_rmse_change"] = float(
                        (out["dev_metrics_level_calibrated"]["rmse"] - rb) / rb)
            except Exception as exc3:
                out["notes"].append("level_calibration_failed_" + type(exc3).__name__)
        if "flat_day" in keep_pred and base is not None:
            bl = 0.5 * (keep_pred["flat_day"] + base)
            out["dev_metrics_blend_day_dayneigh"] = _metrics(bl, yd, gd)
        out["interpretation_limits"] = "single_61_day_development_window_one_city_ranking_measured_on_the_mechanical_selection_metric_confirmation_transfer_untested"
        return out
    except Exception as exc:
        out["fallback_used"] = True
        out["fallback_reason"] = type(exc).__name__
        out["notes"].append("broad_handler_returned_diagnostic_only_no_candidate_comparison_completed")
        return out
