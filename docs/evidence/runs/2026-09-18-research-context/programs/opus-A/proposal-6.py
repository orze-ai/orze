import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

REF = {"max_iter": 250, "learning_rate": 0.08, "max_leaf_nodes": 31,
       "l2_regularization": 1, "random_state": 1729}
SEEDS_A = [1729, 7, 20260918]
FACTOR_LO = 1.0
FACTOR_HI = 1.10
N_FOLDS = 5
CHRONO_BLOCK = 40
TREND_WINDOW = 365
SLOPE_LO = 0.0
SLOPE_HI = 0.0012
MAX_EXTRA = 0.15
W_A = 0.5


def _rmse(a, p):
    d = np.asarray(p, dtype=float) - np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def _fit(Xs, ylog, s):
    par = dict(REF)
    par["random_state"] = int(s)
    m = HistGradientBoostingRegressor(**par)
    m.fit(Xs, ylog)
    return m


def _daily_mean(days, vals):
    u = np.unique(days)
    out = np.empty(u.size, dtype=float)
    for i in range(u.size):
        out[i] = float(np.mean(vals[days == u[i]]))
    return u, out


def _fit_line(days, vals, day_ref, slope_fixed=None):
    u, dm = _daily_mean(days, vals)
    sel = u >= (day_ref - TREND_WINDOW)
    du = u[sel].astype(float)
    dv = dm[sel]
    if du.size < 30:
        return None
    if slope_fixed is None:
        A = np.vstack([np.ones_like(du), du]).T
        coef = np.linalg.lstsq(A, dv, rcond=None)[0]
        raw = float(coef[1])
    else:
        raw = float(slope_fixed)
    slope = float(min(max(raw, SLOPE_LO), SLOPE_HI))
    inter = float(np.mean(dv - slope * du))
    return {"intercept": inter, "slope": slope, "slope_raw": raw, "n_days": int(du.size)}


def _trend(line, t, day_max):
    tt = np.asarray(t, dtype=float)
    v = line["intercept"] + line["slope"] * tt
    vend = line["intercept"] + line["slope"] * float(day_max)
    return np.where(tt > day_max, np.minimum(v, vend + MAX_EXTRA), v)


def _clip_factor(num, den):
    if not np.isfinite(den) or den <= 0.0:
        return None, None
    raw = float(num) / float(den)
    return float(min(max(raw, FACTOR_LO), FACTOR_HI)), raw


def fit_predict(train, inputs, seed):
    names = list(train['feature_names'])
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xev = np.asarray(inputs['X'], dtype=float)
    ei = names.index('elapsed_day')
    drop = set(names.index(c) for c in ('elapsed_day', 'yr') if c in names)
    keep = [j for j in range(len(names)) if j not in drop]
    ylog = np.log1p(np.maximum(y, 0.0))
    day = np.round(X[:, ei]).astype(int)
    tev = np.round(Xev[:, ei]).astype(int)
    day_max = int(day.max())
    dates = np.unique(day)

    findings = {
        "fallback_used": False,
        "branch_policy": "fixed a priori: 0.5/0.5 blend of (A) seed-averaged reference HGB on log1p(cnt) over all raw columns and (B) HGB on log1p(cnt) with elapsed_day and yr removed plus an explicit clipped linear log-trend in elapsed_day; all diagnostics are reported but never switch the branch",
        "dropped_columns_branch_B": sorted([names[j] for j in drop]),
        "n_train_rows": int(X.shape[0]),
        "n_train_dates": int(dates.size),
        "n_pred_rows": int(Xev.shape[0]),
        "elapsed_train_max": float(X[:, ei].max()),
        "elapsed_inputs_min": float(Xev[:, ei].min()),
        "elapsed_inputs_max": float(Xev[:, ei].max()),
        "inputs_beyond_train_elapsed_fraction": float(np.mean(Xev[:, ei] > X[:, ei].max())),
        "slope_clip_per_day": [SLOPE_LO, SLOPE_HI],
        "max_extrapolated_log_increment": MAX_EXTRA,
        "blend_weight_branch_A": W_A,
        "transductive_use": "none for fitting, calibration, trend estimation or branch choice; unlabeled input covariates are used only to emit predictions and for elapsed-day range diagnostics",
    }

    try:
        s0 = int(seed) if seed is not None else 0
    except Exception:
        s0 = 0
    rng = np.random.RandomState(abs(s0) % (2 ** 31 - 1))

    oofA = np.full(X.shape[0], np.nan)
    oofB = np.full(X.shape[0], np.nan)
    foldA = []
    foldB = []
    folds_ok = True
    try:
        perm = rng.permutation(dates.size)
        for k in range(N_FOLDS):
            hold = set(dates[perm[k::N_FOLDS]].tolist())
            mva = np.array([d in hold for d in day], dtype=bool)
            mtr = ~mva
            if int(mva.sum()) < 50 or int(mtr.sum()) < 500:
                folds_ok = False
                break
            mA = _fit(X[mtr], ylog[mtr], 1729)
            pa = mA.predict(X[mva])
            oofA[mva] = pa
            foldA.append(round(_rmse(y[mva], np.maximum(np.expm1(pa), 0.0)), 3))
            mB = _fit(X[mtr][:, keep], ylog[mtr], 1729)
            pb = mB.predict(X[mva][:, keep])
            oofB[mva] = pb
            foldB.append(round(_rmse(y[mva], np.maximum(np.expm1(pb), 0.0)), 3))
    except Exception as exc:
        folds_ok = False
        findings["fold_error"] = type(exc).__name__
    if folds_ok and (not np.all(np.isfinite(oofA)) or not np.all(np.isfinite(oofB))):
        folds_ok = False

    findings["interpolation_random_date_folds"] = {
        "purpose": "estimate retransformation level factors and the residual log-trend; this design measures interpolation, not future-period transfer",
        "rmse_branch_A": foldA,
        "rmse_branch_B_no_trend": foldB,
    }

    factorA = 1.0
    factorB = 1.0
    lineB_oof = None
    if folds_ok:
        fa, ra = _clip_factor(float(y.sum()), float(np.maximum(np.expm1(oofA), 0.0).sum()))
        if fa is not None:
            factorA = fa
        findings["factor_A_raw"] = (round(ra, 5) if ra is not None else None)
        findings["factor_A_applied"] = round(factorA, 5)
        lineB_oof = _fit_line(day, ylog - oofB, day_max)
        if lineB_oof is not None:
            pbo = np.maximum(np.expm1(oofB + _trend(lineB_oof, day, day_max)), 0.0)
            fb, rb = _clip_factor(float(y.sum()), float(pbo.sum()))
            if fb is not None:
                factorB = fb
            findings["factor_B_raw"] = (round(rb, 5) if rb is not None else None)
            findings["factor_B_applied"] = round(factorB, 5)
            findings["oof_trend_slope_raw_per_day"] = round(lineB_oof["slope_raw"], 6)
            findings["oof_trend_slope_applied_per_day"] = round(lineB_oof["slope"], 6)
            findings["oof_trend_slope_clipped"] = bool(lineB_oof["slope_raw"] != lineB_oof["slope"])
            pao = np.maximum(np.expm1(oofA) * factorA, 0.0)
            findings["oof_rmse_branch_A_calibrated"] = round(_rmse(y, pao), 3)
            findings["oof_rmse_branch_B_with_trend"] = round(_rmse(y, np.maximum(pbo * factorB, 0.0)), 3)
            findings["oof_rmse_blend"] = round(_rmse(y, W_A * pao + (1.0 - W_A) * np.maximum(pbo * factorB, 0.0)), 3)
            findings["oof_note"] = "interpolation only; branch B has no elapsed_day/yr inside the fold so these numbers do not measure later-period transfer"
    else:
        findings["fallback_used"] = True
        findings["calibration_note"] = "random-date folds unavailable; retransformation factors set to 1.0 and branch B slope estimated from in-sample residuals"

    try:
        if dates.size > CHRONO_BLOCK + 200:
            cut = dates.size - CHRONO_BLOCK
            trd = set(dates[:cut].tolist())
            mtr = np.array([d in trd for d in day], dtype=bool)
            mva = ~mtr
            dmax_tr = int(day[mtr].max())
            cA = _fit(X[mtr], ylog[mtr], 1729)
            pa = np.maximum(np.expm1(cA.predict(X[mva])) * factorA, 0.0)
            cB = _fit(X[mtr][:, keep], ylog[mtr], 1729)
            rin = ylog[mtr] - cB.predict(X[mtr][:, keep])
            slope_fix = (lineB_oof["slope"] if lineB_oof is not None else None)
            lc = _fit_line(day[mtr], rin, dmax_tr, slope_fixed=slope_fix)
            if lc is not None:
                pb = np.maximum(np.expm1(cB.predict(X[mva][:, keep]) + _trend(lc, day[mva], dmax_tr)) * factorB, 0.0)
            else:
                pb = pa
            pbl = W_A * pa + (1.0 - W_A) * pb
            findings["chronological_last_block_diagnostic"] = {
                "role": "diagnostic_only_not_used_for_selection",
                "n_val_rows": int(mva.sum()),
                "val_day_range": [int(dates[cut]), int(dates[-1])],
                "rmse_branch_A": round(_rmse(y[mva], pa), 3),
                "rmse_branch_B_trend": round(_rmse(y[mva], pb), 3),
                "rmse_blend": round(_rmse(y[mva], pbl), 3),
                "trend_slope_used": (round(lc["slope"], 6) if lc is not None else None),
                "val_actual_mean": round(float(y[mva].mean()), 3),
                "pred_mean_A": round(float(pa.mean()), 3),
                "pred_mean_B": round(float(pb.mean()), 3),
                "caveat": "rising spring block with an in-sample residual level; the parent program recorded that this block mis-ranks candidates relative to actual train->development transfer, so it is reported only",
            }
    except Exception as exc:
        findings["chronological_diagnostic_error"] = type(exc).__name__

    pred = None
    try:
        la = []
        for s in SEEDS_A:
            la.append(_fit(X, ylog, s).predict(Xev))
        predA = np.maximum(np.expm1(np.mean(np.vstack(la), axis=0)) * factorA, 0.0)
        mB = _fit(X[:, keep], ylog, 1729)
        rfull = ylog - mB.predict(X[:, keep])
        slope_fix = (lineB_oof["slope"] if lineB_oof is not None else None)
        lineB = _fit_line(day, rfull, day_max, slope_fixed=slope_fix)
        if lineB is None:
            predB = predA
            findings["fallback_used"] = True
            findings["branch_B_note"] = "trend line unavailable; branch B replaced by branch A predictions"
        else:
            tv = _trend(lineB, tev, day_max)
            predB = np.maximum(np.expm1(mB.predict(Xev[:, keep]) + tv) * factorB, 0.0)
            findings["final_trend_source"] = "slope from out-of-fold residual line when available (clipped), level re-centred on full-model in-sample daily residuals of the last 365 training days"
            findings["final_trend_slope_per_day"] = round(lineB["slope"], 6)
            findings["final_trend_days_used"] = int(lineB["n_days"])
            findings["trend_log_increment_at_max_eval_day"] = round(float(np.max(tv) - (lineB["intercept"] + lineB["slope"] * float(day_max))), 5)
        pred = W_A * predA + (1.0 - W_A) * predB
        findings["selected_branch"] = "blend_0.5_seedavg_hgb_log1p_allcols_plus_0.5_detrended_hgb_with_extrapolated_log_trend"
        findings["final_pred_mean_branch_A"] = round(float(predA.mean()), 3)
        findings["final_pred_mean_branch_B"] = round(float(predB.mean()), 3)
        findings["final_fit"] = "branch A: HistGradientBoostingRegressor(reference settings) on log1p(cnt) over all supplied raw columns, log-space average of 3 seeds, expm1 scaled by clipped factor_A; branch B: same settings without elapsed_day/yr, clipped linear log-trend added before expm1 and scaled by clipped factor_B"
    except Exception as exc:
        findings["final_fit_error"] = type(exc).__name__
        findings["fallback_used"] = True
        findings["selected_branch"] = "raw_target_reference_after_final_fit_error"
        m = HistGradientBoostingRegressor(**REF)
        m.fit(X, y)
        pred = np.maximum(m.predict(Xev), 0.0)

    pred = np.asarray(pred, dtype=float)
    bad = ~np.isfinite(pred)
    if bool(bad.any()):
        findings["nonfinite_predictions_replaced"] = int(bad.sum())
        findings["fallback_used"] = True
        pred = np.where(bad, float(np.median(y)), pred)
    pred = np.maximum(pred, 0.0)
    findings["pred_mean"] = round(float(pred.mean()), 3)
    findings["pred_max"] = round(float(pred.max()), 3)
    findings["train_y_mean"] = round(float(y.mean()), 3)
    return {"prediction": pred.tolist(), "findings": findings}
