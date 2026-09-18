import numpy as np

GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5]
CLIP = 3.0
NPERM = 200


def _pooled_slope(t, y, g, pats):
    num = 0.0
    den = 0.0
    slopes = []
    for p in pats:
        m = (g == p)
        if int(m.sum()) < 3:
            continue
        dt = t[m] - t[m].mean()
        dy = y[m] - y[m].mean()
        sd = float(np.dot(dt, dt))
        if sd <= 0.0:
            continue
        nu = float(np.dot(dt, dy))
        num += nu
        den += sd
        slopes.append(nu / sd)
    beta = (num / den) if den > 0.0 else 0.0
    return float(beta), slopes


def _cv(t, y, g, folds, pats_all, want_group=False):
    sse = dict((lam, 0.0) for lam in GRID)
    pat_mae = dict((lam, []) for lam in GRID)
    n = 0
    for fold in folds:
        m = np.isin(g, np.array(fold))
        if int(m.sum()) == 0 or int((~m).sum()) == 0:
            continue
        tr_pats = [p for p in pats_all if p not in fold]
        beta, _ = _pooled_slope(t[~m], y[~m], g[~m], tr_pats)
        const = float(np.median(y[~m]))
        tc = t[m] - float(t[m].mean())
        yv = y[m]
        gv = g[m]
        n += int(m.sum())
        for lam in GRID:
            pred = const + np.clip(lam * beta * tc, -CLIP, CLIP)
            err = pred - yv
            sse[lam] += float(np.dot(err, err))
            if want_group:
                for p in fold:
                    mm = (gv == p)
                    if int(mm.sum()) > 0:
                        pat_mae[lam].append(float(np.mean(np.abs(err[mm]))))
    out = {}
    for lam in GRID:
        rmse = float(np.sqrt(sse[lam] / max(n, 1)))
        gm = float(np.mean(pat_mae[lam])) if (want_group and pat_mae[lam]) else None
        out[lam] = (rmse, gm)
    return out


def fit_predict(train, inputs, seed):
    findings = {
        "errors": [],
        "fallback_used": False,
        "design": ("deployed prediction = median(train y) + lam*beta*(test_time - mean test_time over "
                    "evaluation rows), clipped to +-3 UPDRS. beta is one pooled patient-demeaned OLS slope "
                    "of motor_UPDRS on test_time (within-patient channel only; between-patient level "
                    "differences removed by demeaning). lam chosen on a fixed grid by grouped 4-fold CV "
                    "over the train patients (7 held-out patients per fold), scored by row RMSE with the "
                    "held-out level set to the fold-training median and the deviation re-centred over the "
                    "held-out rows, i.e. the same operations as deployment. lam=0 reproduces the constant."),
        "transductive": ("evaluation-row test_time values are used only to centre the deviation at zero mean "
                          "so the predicted level equals median(train y); no evaluation labels, groups or row ids"),
        "caveat": ("28 correlated training patients and a 7-patient evaluation cohort are a small group-level "
                    "sample; lam=0 is an informative null, not a failure"),
    }
    try:
        ytr = np.asarray(train["y"], dtype=float)
        Xtr = np.asarray(train["X"], dtype=float)
        Xev = np.asarray(inputs["X"], dtype=float)
        names = [str(v) for v in train["feature_names"]]
        ev_names = [str(v) for v in inputs["feature_names"]]
        const = float(np.median(ytr))
        nev = int(Xev.shape[0])
        if ("test_time" not in names) or ("test_time" not in ev_names):
            findings["fallback_used"] = True
            findings["errors"].append("test_time column absent; deployed constant median(train y) only")
            return {"prediction": [const] * nev, "findings": findings}
        g = np.array([str(v) for v in train["groups"]])
        t = Xtr[:, names.index("test_time")].astype(float)
        tev = Xev[:, ev_names.index("test_time")].astype(float)
        pats_all = sorted(set(g.tolist()))
        nf = 4
        folds = [[] for _ in range(nf)]
        for k, p in enumerate(pats_all):
            folds[k % nf].append(p)
        res = _cv(t, y=ytr, g=g, folds=folds, pats_all=pats_all, want_group=True)
        obj = dict((lam, res[lam][0]) for lam in GRID)
        obj_lam0 = obj[0.0]
        lam_best = min(GRID, key=lambda L: (obj[L], L))
        gain = float(obj_lam0 - obj[lam_best])
        rng = np.random.RandomState(int(seed) if seed is not None else 0)
        ge = 0
        perm_gains = []
        for _ in range(NPERM):
            tp = t.copy()
            for p in pats_all:
                m = (g == p)
                idx = np.where(m)[0]
                tp[idx] = t[idx][rng.permutation(idx.size)]
            rp = _cv(tp, y=ytr, g=g, folds=folds, pats_all=pats_all, want_group=False)
            o0 = rp[0.0][0]
            ob = min(rp[L][0] for L in GRID)
            pg = float(o0 - ob)
            perm_gains.append(pg)
            if pg >= gain - 1e-12:
                ge += 1
        perm_p = float((ge + 1.0) / (NPERM + 1.0))
        beta_all, slopes = _pooled_slope(t, ytr, g, pats_all)
        sl = np.array(slopes, dtype=float) if slopes else np.array([0.0])
        dev = np.clip(lam_best * beta_all * (tev - float(tev.mean())), -CLIP, CLIP)
        pred = const + dev
        findings["constant"] = round(const, 4)
        findings["n_train_patients"] = len(pats_all)
        findings["beta_per_day_pooled"] = round(float(beta_all), 6)
        findings["beta_over_full_test_time_range"] = round(float(beta_all) * float(t.max() - t.min()), 4)
        findings["per_patient_slope"] = {
            "median": round(float(np.median(sl)), 6),
            "mean": round(float(np.mean(sl)), 6),
            "sd": round(float(np.std(sl)), 6),
            "frac_positive": round(float(np.mean(sl > 0.0)), 4),
            "n": int(sl.size),
        }
        findings["cv_grouped_4folds_of_7_patients"] = {
            "grid": GRID,
            "row_rmse": [round(float(obj[L]), 4) for L in GRID],
            "group_mae": [None if res[L][1] is None else round(float(res[L][1]), 4) for L in GRID],
            "lam_selected": lam_best,
            "obj_lam0": round(float(obj_lam0), 4),
            "obj_best": round(float(obj[lam_best]), 4),
            "gain": round(gain, 4),
            "perm_p_within_patient_time_shuffle": perm_p,
            "perm_gain_mean": round(float(np.mean(perm_gains)), 4) if perm_gains else None,
            "n_perm": NPERM,
        }
        findings["prediction_summary"] = {
            "n": nev,
            "mean": round(float(np.mean(pred)), 4),
            "sd": round(float(np.std(pred)), 4),
            "min": round(float(np.min(pred)), 4),
            "max": round(float(np.max(pred)), 4),
        }
        findings["identical_to_constant"] = bool(lam_best == 0.0)
        out = [float(v) for v in pred]
        if (not np.all(np.isfinite(out))) or len(out) != nev:
            findings["fallback_used"] = True
            findings["errors"].append("non-finite or wrong-length prediction; deployed constant instead")
            return {"prediction": [const] * nev, "findings": findings}
        return {"prediction": out, "findings": findings}
    except Exception as exc:
        findings["fallback_used"] = True
        findings["errors"].append("exception: " + type(exc).__name__ + ": " + str(exc)[:300])
        try:
            c = float(np.median(np.asarray(train["y"], dtype=float)))
        except Exception:
            c = 20.0
        return {"prediction": [c] * len(inputs["X"]), "findings": findings}
