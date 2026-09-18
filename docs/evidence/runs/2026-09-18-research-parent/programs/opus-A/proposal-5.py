import json
import math
import numpy as np


def find_col(names, key):
    for i in range(len(names)):
        if names[i] == key:
            return i
    return -1


def group_index(keys):
    order = []
    table = {}
    for i in range(len(keys)):
        k = keys[i]
        if k not in table:
            table[k] = []
            order.append(k)
        table[k].append(i)
    return order, table


def level_set(y, order, table):
    pm = []
    for k in order:
        ii = np.array(table[k], dtype=np.int64)
        pm.append(float(np.mean(y[ii])))
    pma = np.array(pm, dtype=float)
    return {
        "row_mean": float(np.mean(y)),
        "row_median": float(np.median(y)),
        "patient_mean": float(np.mean(pma)),
        "patient_median": float(np.median(pma)),
    }


def pooled_slope(t, y, order, table):
    num = 0.0
    den = 0.0
    slopes = []
    for k in order:
        ii = np.array(table[k], dtype=np.int64)
        if ii.shape[0] < 4:
            continue
        tt = t[ii]
        tc = tt - float(np.mean(tt))
        ss = float(np.sum(tc * tc))
        if ss <= 1e-9:
            continue
        yy = y[ii]
        b = float(np.sum(tc * (yy - float(np.mean(yy)))) / ss)
        slopes.append(b)
        num += ss * b
        den += ss
    if den <= 1e-12:
        return 0.0, slopes
    return num / den, slopes


def fit_predict(train, inputs, seed):
    notes = []
    fn = list(train["feature_names"])
    Xtr = np.asarray(train["X"], dtype=float)
    ytr = np.asarray(train["y"], dtype=float)
    gtr = [str(v) for v in train["groups"]]
    it = find_col(fn, "test_time")
    if it < 0:
        notes.append("fallback: no test_time column in train; time term disabled")
        ttr = np.zeros(Xtr.shape[0], dtype=float)
    else:
        ttr = Xtr[:, it]
    order, table = group_index(gtr)
    lv_keys = ["row_mean", "row_median", "patient_mean", "patient_median"]
    lam_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    sse = {}
    cnt = {}
    pmae = {}
    for lk in lv_keys:
        for lam in lam_grid:
            sse[(lk, lam)] = 0.0
            cnt[(lk, lam)] = 0
            pmae[(lk, lam)] = []
    for p in order:
        rest = []
        for k in order:
            if k != p:
                rest.extend(table[k])
        ridx = np.array(rest, dtype=np.int64)
        yo = ytr[ridx]
        to = ttr[ridx]
        go = [gtr[int(j)] for j in ridx]
        oo, ot = group_index(go)
        lvo = level_set(yo, oo, ot)
        bo, tmp = pooled_slope(to, yo, oo, ot)
        hi = np.array(table[p], dtype=np.int64)
        yh = ytr[hi]
        tc = ttr[hi] - float(np.mean(ttr[hi]))
        for lk in lv_keys:
            for lam in lam_grid:
                err = (lvo[lk] + lam * bo * tc) - yh
                sse[(lk, lam)] += float(np.sum(err * err))
                cnt[(lk, lam)] += int(hi.shape[0])
                pmae[(lk, lam)].append(float(np.mean(np.abs(err))))
    lopo = {}
    for lk in lv_keys:
        for lam in lam_grid:
            n = max(cnt[(lk, lam)], 1)
            lopo[(lk, lam)] = (
                math.sqrt(sse[(lk, lam)] / n),
                float(np.mean(np.array(pmae[(lk, lam)], dtype=float))),
            )
    base = lopo[("row_median", 0.0)]
    best = ("row_median", 0.0)
    for lk in lv_keys:
        for lam in lam_grid:
            cand = lopo[(lk, lam)]
            if cand[1] > 1.03 * base[1]:
                continue
            if cand[0] < lopo[best][0] - 1e-12:
                best = (lk, lam)
    lv = level_set(ytr, order, table)
    bfull, slopes = pooled_slope(ttr, ytr, order, table)
    level = lv[best[0]]
    lam = best[1]
    Xe = np.asarray(inputs["X"], dtype=float)
    fne = list(inputs["feature_names"])
    iae = find_col(fne, "age")
    ise = find_col(fne, "sex")
    ite = find_col(fne, "test_time")
    nrow = Xe.shape[0]
    if ite < 0:
        notes.append("fallback: no test_time in inputs; constant prediction used")
        te = np.zeros(nrow, dtype=float)
        lam = 0.0
    else:
        te = Xe[:, ite]
    center = np.full(nrow, float(np.mean(te)), dtype=float)
    sizes = []
    if lam > 0.0 and iae >= 0 and ise >= 0:
        keys = []
        for i in range(nrow):
            keys.append((round(float(Xe[i, iae]), 6), round(float(Xe[i, ise]), 6)))
        eo, et = group_index(keys)
        small = 0
        for k in eo:
            ii = np.array(et[k], dtype=np.int64)
            sizes.append(int(ii.shape[0]))
            if ii.shape[0] >= 20:
                center[ii] = float(np.mean(te[ii]))
            else:
                small += 1
        if len(eo) < 2:
            notes.append("fallback: transductive (age,sex) grouping gave fewer than 2 clusters; global centering used")
        if small > 0:
            notes.append("fallback: " + str(small) + " (age,sex) clusters had fewer than 20 rows; those rows used global centering")
    elif lam > 0.0:
        notes.append("fallback: age or sex missing in inputs; global centering used")
    pred = level + lam * bfull * (te - center)
    lo = float(np.min(ytr)) - 10.0
    hi2 = float(np.max(ytr)) + 10.0
    pred = np.clip(np.nan_to_num(pred, nan=level, posinf=hi2, neginf=lo), lo, hi2)
    sl = np.array(slopes, dtype=float)
    grid = {}
    for lk in lv_keys:
        for lam2 in lam_grid:
            grid[lk + "|" + str(lam2)] = [round(lopo[(lk, lam2)][0], 4), round(lopo[(lk, lam2)][1], 4)]
    findings = {
        "question": "does within-patient temporal drift of interpolated motor_UPDRS transfer to unseen patients via transductive (age,sex) centering of test_time",
        "chosen_level_estimator": best[0],
        "chosen_lambda": lam,
        "pooled_slope_per_day": round(float(bfull), 6),
        "n_patient_slopes": int(sl.shape[0]),
        "slope_positive_fraction": (round(float(np.mean(sl > 0.0)), 4) if sl.shape[0] > 0 else None),
        "slope_mean": (round(float(np.mean(sl)), 6) if sl.shape[0] > 0 else None),
        "slope_sd": (round(float(np.std(sl)), 6) if sl.shape[0] > 0 else None),
        "lopo_rmse_and_patient_mae_grid": grid,
        "lopo_constant_row_median_reference": [round(base[0], 4), round(base[1], 4)],
        "lopo_relative_rmse_gain_pct": round(100.0 * (base[0] - lopo[best][0]) / max(base[0], 1e-9), 4),
        "eval_cluster_sizes": sizes,
        "eval_cluster_count": len(sizes),
        "prediction_spread": [round(float(np.min(pred)), 4), round(float(np.max(pred)), 4)],
        "fallbacks_used": notes,
        "selection_rule": "leave-one-training-patient-out row RMSE, candidates rejected if LOPO equal-patient MAE exceeds 1.03x constant row-median reference",
        "transductive_use": "unlabeled evaluation rows grouped by exact (age,sex) solely to center test_time within a presumed patient; no evaluation labels, groups or row ids used",
        "caveat": "labels are linearly interpolated clinician ratings, so a fitted within-patient slope reflects interpolation between visits and is not an independent clinical measurement"
    }
    return {"prediction": [float(v) for v in pred], "findings": findings}
