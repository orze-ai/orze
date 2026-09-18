import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

REF = {"max_iter": 250, "learning_rate": 0.08, "max_leaf_nodes": 31,
       "l2_regularization": 1, "random_state": 1729}


def _rmse(a, p):
    d = np.asarray(p, dtype=float) - np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def _mae(a, p):
    return float(np.mean(np.abs(np.asarray(p, dtype=float) - np.asarray(a, dtype=float))))


def _dmae(a, p, days):
    vals = []
    for u in np.unique(days):
        m = days == u
        vals.append(float(np.mean(np.abs(p[m] - a[m]))))
    return float(np.mean(vals))


def _trend(el, y):
    days = np.round(el).astype(int)
    ud = np.unique(days)
    m = np.array([float(y[days == u].mean()) for u in ud])
    lm = np.log(np.maximum(m, 1.0))
    t = ud.astype(float)
    cols = [np.ones_like(t), t]
    for h in (1.0, 2.0):
        ang = 2.0 * np.pi * h * t / 365.25
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    A = np.vstack(cols).T
    coef = np.linalg.lstsq(A, lm, rcond=None)[0]
    return float(coef[1]), float(t.max()), float(t.mean())


def _g(el, b, dmax, dref, kappa):
    eff = np.where(el <= dmax, el, dmax + kappa * (el - dmax))
    return np.exp(b * (eff - dref))


def _cands(Xtr, ytr, eltr, Xev, elev, keep, notes):
    out = {}
    Xtrn = Xtr[:, keep]
    Xevn = Xev[:, keep]

    def add(name, fn):
        try:
            p = np.asarray(fn(), dtype=float)
            p = np.where(np.isfinite(p), p, 0.0)
            out[name] = np.maximum(p, 0.0)
        except Exception as exc:
            notes[name + "_error"] = type(exc).__name__

    add("hgb_raw", lambda: HistGradientBoostingRegressor(**REF).fit(Xtr, ytr).predict(Xev))
    add("hgb_log", lambda: np.expm1(HistGradientBoostingRegressor(**REF).fit(Xtr, np.log1p(ytr)).predict(Xev)))
    add("hgb_poisson", lambda: HistGradientBoostingRegressor(loss="poisson", **REF).fit(Xtr, ytr).predict(Xev))
    add("hgb_noelapsed", lambda: HistGradientBoostingRegressor(**REF).fit(Xtrn, ytr).predict(Xevn))
    try:
        b, dmax, dref = _trend(eltr, ytr)
        notes["log_slope_per_day"] = round(b, 6)

        def mk(kap):
            def f():
                g = np.maximum(_g(eltr, b, dmax, dref, kap), 1e-6)
                m = HistGradientBoostingRegressor(**REF)
                m.fit(Xtrn, ytr / g, sample_weight=g * g)
                return m.predict(Xevn) * _g(elev, b, dmax, dref, kap)
            return f

        add("trend_k0", mk(0.0))
        add("trend_k1", mk(1.0))
    except Exception as exc:
        notes["trend_setup_error"] = type(exc).__name__
    if "hgb_raw" in out and "hgb_log" in out:
        out["blend_raw_log"] = 0.5 * (out["hgb_raw"] + out["hgb_log"])
    if "hgb_raw" in out and "hgb_poisson" in out:
        out["blend_raw_pois"] = 0.5 * (out["hgb_raw"] + out["hgb_poisson"])
    return out


def _score(preds, a, days):
    res = {}
    for k in sorted(preds):
        p = preds[k]
        den = float(np.dot(p, p))
        sc = float(np.dot(p, a) / den) if den > 1e-9 else 1.0
        res[k] = {"rmse": round(_rmse(a, p), 3), "mae": round(_mae(a, p), 3),
                  "date_mae": round(_dmae(a, p, days), 3),
                  "opt_scale": round(sc, 4),
                  "rmse_opt_scale": round(_rmse(a, sc * p), 3),
                  "pred_mean": round(float(p.mean()), 2)}
    return res


def analyze(data, history, seed):
    f = {"role": "diagnostic_only_no_prediction",
         "transductive_use": "none; only supplied train and development arrays are used"}
    tr = data["train"]
    dv = data["development"]
    names = list(tr["feature_names"])
    Xtr = np.asarray(tr["X"], dtype=float)
    ytr = np.asarray(tr["y"], dtype=float)
    Xdv = np.asarray(dv["X"], dtype=float)
    ydv = np.asarray(dv["y"], dtype=float)
    ei = names.index("elapsed_day")
    hi = names.index("hr")
    wsi = names.index("weathersit")
    keep = [j for j in range(len(names)) if j != ei]
    eltr = Xtr[:, ei]
    eldv = Xdv[:, ei]
    dtr = np.round(eltr).astype(int)
    ddv = np.round(eldv).astype(int)
    f["n_train_rows"] = int(Xtr.shape[0])
    f["n_dev_rows"] = int(Xdv.shape[0])
    f["train_day_range"] = [int(dtr.min()), int(dtr.max())]
    f["dev_day_range"] = [int(ddv.min()), int(ddv.max())]
    f["train_y_mean"] = round(float(ytr.mean()), 2)
    f["dev_y_mean"] = round(float(ydv.mean()), 2)
    last = dtr >= (dtr.max() - 60)
    f["train_last61d_y_mean"] = round(float(ytr[last].mean()), 2)
    f["dev_over_train_last61_ratio"] = round(float(ydv.mean() / max(float(ytr[last].mean()), 1e-9)), 4)
    try:
        b0, dmax0, dref0 = _trend(eltr, ytr)
        gap = float(np.mean(ddv)) - float(np.mean(dtr[last]))
        f["trend_gap_days"] = round(gap, 1)
        f["trend_implied_ratio_over_gap"] = round(float(np.exp(b0 * gap)), 4)
    except Exception as exc:
        f["trend_diag_error"] = type(exc).__name__
    notes1 = {}
    p1 = _cands(Xtr, ytr, eltr, Xdv, eldv, keep, notes1)
    f["future_transfer_train_to_development"] = _score(p1, ydv, ddv)
    f["p1_notes"] = notes1
    ud = np.unique(dtr)
    if int(ud.size) > 200:
        cut = int(ud[-61])
        mi = dtr < cut
        mo = dtr >= cut
        notes2 = {}
        p2 = _cands(Xtr[mi], ytr[mi], eltr[mi], Xtr[mo], eltr[mo], keep, notes2)
        f["internal_last61_train_holdout"] = _score(p2, ytr[mo], dtr[mo])
        f["p2_notes"] = notes2
        try:
            t1 = f["future_transfer_train_to_development"]
            t2 = f["internal_last61_train_holdout"]
            o1 = sorted(t1, key=lambda k: t1[k]["rmse"])
            o2 = sorted(t2, key=lambda k: t2[k]["rmse"])
            f["rank_dev"] = o1
            f["rank_internal"] = o2
            f["internal_top_is_dev_top"] = bool(o1[0] == o2[0])
        except Exception as exc:
            f["rank_error"] = type(exc).__name__
    else:
        f["holdout_note"] = "insufficient training dates for internal 61-day block"
    wk = (ddv - ddv.min()) // 7
    uw = np.unique(wk)
    if "hgb_raw" in p1:
        r = p1["hgb_raw"] - ydv
        hrs = np.round(Xdv[:, hi]).astype(int)
        f["dev_hour_mean_resid_hgb_raw"] = {str(int(h)): round(float(r[hrs == h].mean()), 1) for h in np.unique(hrs)}
        f["dev_hour_actual_mean"] = {str(int(h)): round(float(ydv[hrs == h].mean()), 1) for h in np.unique(hrs)}
        ws = np.round(Xdv[:, wsi]).astype(int)
        f["dev_weathersit_resid_count_hgb_raw"] = {str(int(w)): [round(float(r[ws == w].mean()), 1), int((ws == w).sum())] for w in np.unique(ws)}
        f["dev_week_actual_mean"] = [round(float(ydv[wk == u].mean()), 1) for u in uw]
        f["dev_week_pred_mean_hgb_raw"] = [round(float(p1["hgb_raw"][wk == u].mean()), 1) for u in uw]
    if "trend_k0" in p1:
        rt = p1["trend_k0"] - ydv
        hrs2 = np.round(Xdv[:, hi]).astype(int)
        f["dev_week_pred_mean_trend_k0"] = [round(float(p1["trend_k0"][wk == u].mean()), 1) for u in uw]
        f["dev_hour_mean_resid_trend_k0"] = {str(int(h)): round(float(rt[hrs2 == h].mean()), 1) for h in np.unique(hrs2)}
    try:
        f["history_programs_seen"] = int(len(history))
    except Exception as exc:
        f["history_len_error"] = type(exc).__name__
    return f
