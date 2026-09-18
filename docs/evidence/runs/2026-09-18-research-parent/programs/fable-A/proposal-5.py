import math
import json
import numpy as np


def fit_predict(train, inputs, seed):
    fallbacks = []
    X = np.asarray(train["X"], dtype=float)
    y = np.asarray(train["y"], dtype=float)
    g = [str(v) for v in train["groups"]]
    fn = list(train["feature_names"])
    if "test_time" in fn:
        ti = fn.index("test_time")
    else:
        ti = 2
        fallbacks.append("test_time_not_in_feature_names_used_column_2")
    Xe = np.asarray(inputs["X"], dtype=float)
    t = X[:, ti]
    te = Xe[:, ti]
    n = len(y)
    pats = sorted(set(g))
    idx = {p: [i for i in range(n) if g[i] == p] for p in pats}

    def fit(pat_list):
        # constant = mean of patient means (equal-patient weighting)
        pm = np.array([y[idx[p]].mean() for p in pat_list])
        c = float(pm.mean())
        num = 0.0
        den = 0.0
        tsum = 0.0
        cnt = 0
        for p in pat_list:
            ii = idx[p]
            yd = y[ii] - y[ii].mean()
            td = t[ii] - t[ii].mean()
            num += float((td * yd).sum())
            den += float((td * td).sum())
            tsum += float(t[ii].sum())
            cnt += len(ii)
        slope = num / den if den > 0 else 0.0
        tbar = tsum / cnt if cnt > 0 else 0.0
        return c, slope, tbar

    c_all, slope_all, tbar_all = fit(pats)

    # within-patient R^2 of the pooled slope on train
    ss_w = 0.0
    ss_res = 0.0
    per_slopes = []
    for p in pats:
        ii = idx[p]
        yd = y[ii] - y[ii].mean()
        td = t[ii] - t[ii].mean()
        ss_w += float((yd * yd).sum())
        ss_res += float(((yd - slope_all * td) ** 2).sum())
        d = float((td * td).sum())
        per_slopes.append(float((td * yd).sum() / d) if d > 0 else 0.0)
    within_r2 = 1.0 - ss_res / ss_w if ss_w > 0 else 0.0

    # leave-one-patient-out on train: constant alone vs constant + slope
    def lopo(use_slope):
        se = 0.0
        cnt = 0
        pe = []
        for p in pats:
            others = [q for q in pats if q != p]
            c, s, tb = fit(others)
            ho = idx[p]
            pred = np.full(len(ho), c)
            if use_slope:
                pred = pred + s * (t[ho] - tb)
            err = y[ho] - pred
            se += float((err ** 2).sum())
            cnt += len(ho)
            pe.append(float(np.abs(err).mean()))
        return math.sqrt(se / cnt), float(np.mean(pe))

    rm_c, gm_c = lopo(False)
    rm_s, gm_s = lopo(True)

    pred = c_all + slope_all * (te - tbar_all)
    lo = float(y.min())
    hi = float(y.max())
    n_clip = int(((pred < lo) | (pred > hi)).sum())
    pred = np.clip(pred, lo, hi)
    if not np.all(np.isfinite(pred)):
        fallbacks.append("nonfinite_prediction_replaced_by_constant")
        pred = np.where(np.isfinite(pred), pred, c_all)

    findings = {
        "method": "constant (mean of train patient means) + pooled within-patient fixed-effects slope on test_time; no voice features; no transductive use of inputs",
        "constant": round(c_all, 4),
        "slope_per_day": round(slope_all, 6),
        "train_tbar": round(tbar_all, 3),
        "within_patient_r2_of_pooled_slope": round(within_r2, 4),
        "per_patient_slope_quantiles": [round(float(q), 5) for q in np.percentile(per_slopes, [10, 25, 50, 75, 90])],
        "train_lopo_28": {
            "const": {"row_rmse": round(rm_c, 4), "equal_patient_mae": round(gm_c, 4)},
            "const_plus_slope": {"row_rmse": round(rm_s, 4), "equal_patient_mae": round(gm_s, 4)},
            "row_rmse_delta_slope_minus_const": round(rm_s - rm_c, 4),
        },
        "eval_test_time_range": [round(float(te.min()), 2), round(float(te.max()), 2)],
        "eval_pred_range": [round(float(pred.min()), 3), round(float(pred.max()), 3)],
        "n_clipped_to_train_range": n_clip,
        "fallback_used": bool(fallbacks),
        "fallbacks": fallbacks,
        "note": "one-factor change vs a constant predictor: only the test_time trend term is added; dev labels are not used",
    }
    s = json.dumps(findings)
    if len(s.encode("utf-8")) > 8000:
        findings = {"constant": round(c_all, 4), "slope_per_day": round(slope_all, 6), "fallback_used": bool(fallbacks)}
    return {"prediction": [float(v) for v in pred], "findings": findings}
