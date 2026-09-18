import numpy as np

GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0]
CLIP = 3.0
RIDGE = 1e-3
NPERM = 200


def _agg(X, y, g):
    gs = [str(v) for v in g]
    ga = np.asarray(gs)
    keys = sorted(set(gs))
    P = []
    ym = []
    nn = []
    for k in keys:
        ii = np.where(ga == k)[0]
        P.append(np.median(X[ii], axis=0))
        ym.append(float(np.mean(y[ii])))
        nn.append(float(len(ii)))
    return keys, np.asarray(P, dtype=float), np.asarray(ym, dtype=float), np.asarray(nn, dtype=float)


def _wfit(Pc, yc, w):
    A = Pc * w[:, None]
    G = Pc.T @ A + RIDGE * np.eye(Pc.shape[1])
    return np.linalg.solve(G, A.T @ yc)


def _lopo(P, Y, nn, cols):
    m = P.shape[0]
    d = np.zeros(m)
    c = np.zeros(m)
    idx = np.arange(m)
    for j in range(m):
        tr = idx[idx != j]
        Pt = P[np.ix_(tr, cols)]
        w = nn[tr]
        mu = np.average(Pt, axis=0, weights=w)
        var = np.average((Pt - mu) ** 2, axis=0, weights=w)
        sd = np.sqrt(var) + 1e-9
        yb = float(np.average(Y[tr], weights=w))
        b = _wfit((Pt - mu) / sd, Y[tr] - yb, w)
        d[j] = float(((P[j, cols] - mu) / sd) @ b)
        c[j] = yb
    return d, c


def _obj(Y, nn, d, c, lam):
    r = Y - c - lam * d
    return float(np.sqrt(np.sum(nn * r * r) / np.sum(nn)))


def _eval_set(P, Y, nn, cols, rng, nperm):
    d, c = _lopo(P, Y, nn, cols)
    objs = [_obj(Y, nn, d, c, l) for l in GRID]
    k = int(np.argmin(objs))
    gain = objs[0] - objs[k]
    worse = 0
    for _ in range(nperm):
        Yp = Y[rng.permutation(len(Y))]
        dp, cp = _lopo(P, Yp, nn, cols)
        op = [_obj(Yp, nn, dp, cp, l) for l in GRID]
        if (op[0] - min(op)) >= gain - 1e-12:
            worse += 1
    p = (worse + 1.0) / (nperm + 1.0)
    res = {"lam": GRID[k], "obj_lam0": round(objs[0], 4), "obj_best": round(objs[k], 4),
           "gain": round(gain, 4), "perm_p": round(p, 4),
           "objs": [round(o, 4) for o in objs]}
    return res, d, c, cols


def fit_predict(train, inputs, seed):
    n_eval = len(inputs['X'])
    findings = {"errors": [], "fallback_used": False,
                "design": "pre-declared: deployed prediction = median(train y) (same constant as measured leader idea-baseline-0) + lam * age-only patient-level slope, with the deviation term re-centred to mean zero over the unlabeled evaluation rows and clipped to +-3 UPDRS; lam selected on a fixed grid by leave-one-patient-out over the 28 train patients; age_sex, voice16 and age_sex_voice sets are computed under the identical protocol and reported as diagnostics only, they never switch the deployed output",
                "caveat": "LOPO over 28 correlated training patients is optimistic and a 7-patient development cohort is a small group-level sample; a lam of 0 is an informative null, not a failure"}
    try:
        X = np.asarray(train['X'], dtype=float)
        y = np.asarray(train['y'], dtype=float)
        g = np.asarray(train['groups'])
        names = [str(s) for s in train['feature_names']]
        low = [s.lower() for s in names]
        Xe = np.asarray(inputs['X'], dtype=float)
        c0 = float(np.median(y))
        i_age = low.index('age')
        i_sex = low.index('sex')
        i_tt = low.index('test_time')
        voice = [i for i in range(len(names)) if i not in (i_age, i_sex, i_tt)]
        keys, P, Y, nn = _agg(X, y, g)
        rng = np.random.default_rng(int(seed) if seed is not None else 0)
        sets = [("age", [i_age]), ("age_sex", [i_age, i_sex]), ("voice16", voice),
                ("age_sex_voice", [i_age, i_sex] + voice)]
        diag = {}
        dep_cols = [i_age]
        for nm, cols in sets:
            res, d, c, cc = _eval_set(P, Y, nn, cols, rng, NPERM)
            diag[nm] = res
            if nm == "age":
                dep_cols = cc
        lam = float(diag["age"]["lam"])
        w = nn
        mu = np.average(P[:, dep_cols], axis=0, weights=w)
        sd = np.sqrt(np.average((P[:, dep_cols] - mu) ** 2, axis=0, weights=w)) + 1e-9
        yb = float(np.average(Y, weights=w))
        b = _wfit((P[:, dep_cols] - mu) / sd, Y - yb, w)
        Z = (Xe[:, dep_cols] - mu) / sd
        raw = Z @ b
        dev = lam * (raw - float(np.mean(raw)))
        dev = np.clip(dev, -CLIP, CLIP)
        pred = c0 + dev
        if pred.shape[0] != n_eval or not bool(np.all(np.isfinite(pred))):
            raise ValueError("non-finite or wrong-length prediction")
        findings["constant"] = round(c0, 4)
        findings["lambda_selected"] = lam
        findings["slope_per_sd_of_deployed_cols"] = [round(float(v), 4) for v in b]
        findings["deployed_cols"] = [names[i] for i in dep_cols]
        findings["diagnostics_lopo_28_patients"] = diag
        findings["n_train_patients"] = int(len(keys))
        findings["sd_of_patient_mean_motor_updrs"] = round(float(np.std(Y)), 4)
        findings["prediction_summary"] = {"n": int(n_eval), "mean": round(float(np.mean(pred)), 4),
                                          "sd": round(float(np.std(pred)), 4),
                                          "min": round(float(np.min(pred)), 4),
                                          "max": round(float(np.max(pred)), 4)}
        findings["transductive"] = "evaluation-row covariate values are used only to centre the deviation term at zero mean so the predicted level equals median(train y); no evaluation labels, groups or row ids are used"
        return {"prediction": [float(v) for v in pred], "findings": findings}
    except Exception as e:
        findings["fallback_used"] = True
        findings["errors"].append(repr(e)[:400])
        try:
            c0 = float(np.median(np.asarray(train['y'], dtype=float)))
        except Exception:
            c0 = 20.27
        findings["fallback_description"] = "intended age-shrinkage method failed; emitted the train-median constant instead, this is NOT the intended method"
        return {"prediction": [c0] * n_eval, "findings": findings}
