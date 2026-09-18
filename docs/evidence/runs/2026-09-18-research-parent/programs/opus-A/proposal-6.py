import json
import numpy as np

ORDER = ["row_median", "row_mean", "row_huber", "patient_mean_of_means", "patient_median_of_medians", "mid_row_mean_median"]


def _huber_center(y, iters=100):
    y = np.asarray(y, dtype=float)
    m = float(np.median(y))
    mad = float(np.median(np.abs(y - m)))
    s = 1.4826 * mad
    if (not np.isfinite(s)) or s <= 1e-9:
        return m
    d = 1.345 * s
    for _ in range(iters):
        r = y - m
        w = np.ones_like(r)
        big = np.abs(r) > d
        if np.any(big):
            w[big] = d / np.abs(r[big])
        sw = float(np.sum(w))
        if sw <= 0.0:
            return float(np.median(y))
        new = float(np.sum(w * y) / sw)
        if abs(new - m) < 1e-11:
            return new
        m = new
    return m


def _constants(y, g):
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    uq = np.unique(g)
    pmean = np.array([float(np.mean(y[g == u])) for u in uq], dtype=float)
    pmed = np.array([float(np.median(y[g == u])) for u in uq], dtype=float)
    c = {}
    c["row_mean"] = float(np.mean(y))
    c["row_median"] = float(np.median(y))
    c["row_huber"] = float(_huber_center(y))
    c["patient_mean_of_means"] = float(np.mean(pmean))
    c["patient_median_of_medians"] = float(np.median(pmed))
    c["mid_row_mean_median"] = 0.5 * (c["row_mean"] + c["row_median"])
    return c


def _lopo(y, g):
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    uq = np.unique(g)
    se = dict((k, 0.0) for k in ORDER)
    pm = dict((k, []) for k in ORDER)
    n = 0
    for u in uq:
        te = (g == u)
        tr = ~te
        if int(np.sum(tr)) < 2 or int(np.sum(te)) < 1:
            continue
        c = _constants(y[tr], g[tr])
        yt = y[te]
        n += int(yt.size)
        for k in ORDER:
            r = yt - c[k]
            se[k] += float(np.sum(r * r))
            pm[k].append(float(np.mean(np.abs(r))))
    grid = {}
    for k in ORDER:
        if n > 0 and len(pm[k]) > 0:
            grid[k] = [round(float(np.sqrt(se[k] / n)), 4), round(float(np.mean(pm[k])), 4)]
    return grid, int(uq.size), int(n)


def fit_predict(train, inputs, seed=0):
    fallbacks = []
    y = np.asarray(train["y"], dtype=float)
    if train.get("groups") is None:
        g = np.zeros(y.size, dtype=int)
        fallbacks.append("groups_missing_single_pseudo_patient")
    else:
        g = np.asarray(train["groups"])
    n_eval = len(inputs["X"])
    grid = {}
    n_pat = 0
    n_lopo = 0
    chosen = "row_median"
    rejected = []
    try:
        grid, n_pat, n_lopo = _lopo(y, g)
        if "row_median" not in grid:
            raise ValueError("lopo_grid_incomplete")
        ref_mae = grid["row_median"][1]
        best = None
        for k in ORDER:
            if k not in grid:
                continue
            if grid[k][1] > 1.03 * ref_mae:
                rejected.append(k)
                continue
            if best is None or grid[k][0] < grid[best][0] - 1e-12:
                best = k
        if best is None:
            fallbacks.append("all_candidates_rejected_by_equal_patient_mae_guard")
            chosen = "row_median"
        else:
            chosen = best
    except Exception as exc:
        fallbacks.append("lopo_selection_failed_" + type(exc).__name__)
        chosen = "row_median"
    try:
        consts = _constants(y, g)
    except Exception as exc:
        fallbacks.append("full_constants_failed_" + type(exc).__name__)
        consts = {"row_median": float(np.median(y))}
        chosen = "row_median"
    value = float(consts.get(chosen, float(np.median(y))))
    if not np.isfinite(value):
        fallbacks.append("non_finite_constant_replaced_by_row_median")
        value = float(np.median(y))
        chosen = "row_median"
    pred = [value] * n_eval
    gain = None
    try:
        if chosen in grid and "row_median" in grid:
            gain = round(100.0 * (grid["row_median"][0] - grid[chosen][0]) / grid["row_median"][0], 4)
    except Exception:
        gain = None
    findings = {
        "question": "which constant central tendency of interpolated motor_UPDRS transfers to unseen patients, and does LOPO ordering among constants predict held-out-patient ordering",
        "model_class": "constant_only_no_features_no_transduction",
        "transductive_use": "none; unlabeled evaluation X is used only for its row count",
        "selection_rule": "leave-one-training-patient-out row RMSE over a fixed 6-candidate constant list; candidate rejected if its LOPO equal-patient MAE exceeds 1.03x the row-median constant",
        "lopo_rmse_and_patient_mae_grid": grid,
        "lopo_patients": n_pat,
        "lopo_scored_rows": n_lopo,
        "candidate_constants_on_full_train": dict((k, round(float(v), 4)) for k, v in consts.items()),
        "chosen_constant_name": chosen,
        "chosen_constant_value": round(value, 4),
        "rejected_by_patient_mae_guard": rejected,
        "lopo_relative_rmse_gain_vs_row_median_pct": gain,
        "prediction_spread": [round(value, 4), round(value, 4)],
        "eval_rows": int(n_eval),
        "fallbacks_used": fallbacks,
        "caveat": "labels are linearly interpolated clinician ratings; these constants summarise that interpolated label distribution and carry no voice information. A LOPO advantage over 28 training patients need not transfer to 7 held-out patients, and the development/confirmation groups are small patient-level samples."
    }
    try:
        json.dumps(findings)
    except Exception:
        findings = {"chosen_constant_name": chosen, "chosen_constant_value": round(value, 4), "fallbacks_used": fallbacks + ["findings_not_json_serialisable"]}
    return {"prediction": pred, "findings": findings}
