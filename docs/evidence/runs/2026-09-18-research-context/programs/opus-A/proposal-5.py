import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

REF = {"max_iter": 250, "learning_rate": 0.08, "max_leaf_nodes": 31,
       "l2_regularization": 1, "random_state": 1729}

FACTOR_LO = 1.0
FACTOR_HI = 1.10
N_FOLDS = 5
CHRONO_BLOCK = 40


def _rmse(a, p):
    d = np.asarray(p, dtype=float) - np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def _fit_log(X, y):
    m = HistGradientBoostingRegressor(**REF)
    m.fit(X, np.log1p(np.maximum(y, 0.0)))
    return m


def _pred_log(m, X, factor=1.0):
    p = np.expm1(m.predict(X)) * float(factor)
    return np.maximum(p, 0.0)


def _fit_raw(X, y):
    m = HistGradientBoostingRegressor(**REF)
    m.fit(X, y)
    return m


def _pred_raw(m, X):
    return np.maximum(m.predict(X), 0.0)


def fit_predict(train, inputs, seed):
    names = list(train['feature_names'])
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xev = np.asarray(inputs['X'], dtype=float)
    ei = names.index('elapsed_day')
    day = np.round(X[:, ei]).astype(int)
    dates = np.unique(day)
    findings = {
        "fallback_used": False,
        "branch_policy": "fixed a priori: reference HistGradientBoosting on log1p(cnt) over all supplied raw columns; chronological diagnostics are reported but never used to switch branch",
        "n_train_rows": int(X.shape[0]),
        "n_train_dates": int(dates.size),
        "n_pred_rows": int(Xev.shape[0]),
        "elapsed_train_max": float(X[:, ei].max()),
        "elapsed_inputs_min": float(Xev[:, ei].min()),
        "elapsed_inputs_max": float(Xev[:, ei].max()),
        "inputs_beyond_train_elapsed_fraction": float(np.mean(Xev[:, ei] > X[:, ei].max())),
        "transductive_use": "none for fitting, calibration or branch choice; unlabeled input covariates are used only to emit predictions and for range diagnostics",
    }

    try:
        s = int(seed) if seed is not None else 0
    except Exception:
        s = 0
    rng = np.random.RandomState(abs(s) % (2 ** 31 - 1))

    sum_y = 0.0
    sum_p = 0.0
    interp_log = []
    interp_raw = []
    interp_ok = True
    try:
        perm = rng.permutation(dates.size)
        for k in range(N_FOLDS):
            hold = set(dates[perm[k::N_FOLDS]].tolist())
            mva = np.array([d in hold for d in day], dtype=bool)
            mtr = ~mva
            if int(mva.sum()) < 50 or int(mtr.sum()) < 500:
                interp_ok = False
                break
            ml = _fit_log(X[mtr], y[mtr])
            pl = _pred_log(ml, X[mva], 1.0)
            sum_y += float(y[mva].sum())
            sum_p += float(pl.sum())
            interp_log.append(round(_rmse(y[mva], pl), 3))
            mr = _fit_raw(X[mtr], y[mtr])
            interp_raw.append(round(_rmse(y[mva], _pred_raw(mr, X[mva])), 3))
    except Exception as exc:
        interp_ok = False
        findings["interpolation_fold_error"] = type(exc).__name__

    findings["interpolation_random_date_folds"] = {
        "purpose": "estimate expm1 retransformation level bias only; this design measures interpolation, not future-period transfer",
        "rmse_log_target": interp_log,
        "rmse_raw_target": interp_raw,
    }

    factor = 1.0
    if interp_ok and len(interp_log) == N_FOLDS and sum_p > 0.0:
        raw_factor = sum_y / sum_p
        factor = float(min(max(raw_factor, FACTOR_LO), FACTOR_HI))
        findings["retransformation_factor_raw"] = round(float(raw_factor), 5)
        findings["retransformation_factor_applied"] = round(factor, 5)
        findings["retransformation_factor_clipped"] = bool(
            raw_factor < FACTOR_LO or raw_factor > FACTOR_HI)
    else:
        findings["retransformation_factor_raw"] = None
        findings["retransformation_factor_applied"] = 1.0
        findings["retransformation_note"] = "interpolation folds unavailable; uncorrected expm1 used"
        findings["fallback_used"] = True

    try:
        if dates.size > CHRONO_BLOCK + 200:
            cut = dates.size - CHRONO_BLOCK
            trd = set(dates[:cut].tolist())
            vad = set(dates[cut:].tolist())
            mtr = np.array([d in trd for d in day], dtype=bool)
            mva = np.array([d in vad for d in day], dtype=bool)
            ml = _fit_log(X[mtr], y[mtr])
            pl = _pred_log(ml, X[mva], factor)
            mr = _fit_raw(X[mtr], y[mtr])
            pr = _pred_raw(mr, X[mva])
            findings["chronological_last_block_diagnostic"] = {
                "role": "diagnostic_only_not_used_for_selection",
                "n_val_rows": int(mva.sum()),
                "val_day_range": [int(dates[cut]), int(dates[-1])],
                "rmse_log_target": round(_rmse(y[mva], pl), 3),
                "rmse_raw_target": round(_rmse(y[mva], pr), 3),
                "val_actual_mean": round(float(y[mva].mean()), 3),
                "val_pred_mean_log_target": round(float(pl.mean()), 3),
                "val_pred_mean_raw_target": round(float(pr.mean()), 3),
                "caveat": "this block is the rising spring season, so level bias here mixes seasonal rise with growth and previously mis-ranked candidates relative to actual train->development transfer",
            }
    except Exception as exc:
        findings["chronological_diagnostic_error"] = type(exc).__name__

    try:
        mdl = _fit_log(X, y)
        pred = _pred_log(mdl, Xev, factor)
        findings["selected_branch"] = "hgb_log1p_with_interpolation_factor"
        findings["final_fit"] = ("HistGradientBoostingRegressor(reference settings) on log1p(cnt), all supplied raw columns "
                                 "including elapsed_day, inverse expm1 scaled by the clipped interpolation factor")
    except Exception as exc:
        findings["final_fit_error"] = type(exc).__name__
        findings["fallback_used"] = True
        findings["selected_branch"] = "raw_target_reference_after_final_fit_error"
        mdl = _fit_raw(X, y)
        pred = _pred_raw(mdl, Xev)

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
