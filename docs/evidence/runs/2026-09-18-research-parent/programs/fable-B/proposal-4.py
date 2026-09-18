import numpy as np

def _find(names, keys):
    ln = [str(n).lower() for n in names]
    for k in keys:
        for i, n in enumerate(ln):
            if k in n:
                return i
    return None

def _ridge(A, t, alpha):
    mu = A.mean(0)
    tm = float(t.mean())
    Ac = A - mu
    tc = t - tm
    w = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(Ac.shape[1]), Ac.T @ tc)
    return w, tm - float(mu @ w)

def fit_predict(train, inputs, seed):
    names = list(train['feature_names'])
    X = np.asarray(train['X'], float)
    y = np.asarray(train['y'], float)
    g = np.asarray([str(v) for v in train['groups']])
    Xe = np.asarray(inputs['X'], float)
    med = float(np.median(y))
    ia = _find(names, ['age'])
    isx = _find(names, ['sex'])
    it = _find(names, ['test_time', 'time'])
    if ia is None or isx is None or it is None or len({ia, isx, it}) < 3:
        return {'prediction': [med] * len(Xe), 'findings': {'fallback_used': True, 'reason': 'column lookup failed; train median returned', 'feature_names': [str(n) for n in names]}}
    voice = [j for j in range(len(names)) if j not in (ia, isx, it)]
    logcols = [j for j in voice if float(np.min(X[:, j])) > 0]
    def tf(A):
        B = np.array(A, dtype=float, copy=True)
        for j in logcols:
            B[:, j] = np.log(np.maximum(B[:, j], 1e-12))
        return B
    Xt = tf(X)
    Xet = tf(Xe)
    bcols = [ia, isx] + voice
    pats = sorted(set(g.tolist()))
    masks = [g == p for p in pats]
    P = np.array([Xt[m][:, bcols].mean(0) for m in masks])
    Py = np.array([float(y[m].mean()) for m in masks])
    nums = []
    dens = []
    for m in masks:
        tc = X[m, it] - X[m, it].mean()
        yc = y[m] - y[m].mean()
        nums.append(float(tc @ yc))
        dens.append(float(tc @ tc))
    nums = np.array(nums)
    dens = np.array(dens)
    alphas = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    sse = {}
    gerr = {}
    nrow = 0
    def add(key, pred, ytrue):
        e = pred - ytrue
        sse[key] = sse.get(key, 0.0) + float(e @ e)
        gerr.setdefault(key, []).append(float(np.mean(np.abs(e))))
    n_p = len(pats)
    for i in range(n_p):
        m = masks[i]
        oth = np.array([k != i for k in range(n_p)])
        yo = y[~m]
        med_o = float(np.median(yo))
        dsum = float(dens[oth].sum())
        slope_o = float(nums[oth].sum() / dsum) if dsum > 0 else 0.0
        tbar_o = float(X[~m, it].mean())
        mu = P[oth].mean(0)
        sd = P[oth].std(0)
        sd[sd < 1e-12] = 1.0
        A = (P[oth] - mu) / sd
        R = (Xt[m][:, bcols] - mu) / sd
        tdev = slope_o * (X[m, it] - tbar_o)
        yt = y[m]
        add('median', np.full(yt.shape, med_o), yt)
        add('median+time', med_o + tdev, yt)
        for a in alphas:
            w, b = _ridge(A, Py[oth], a)
            pb = R @ w + b
            add('between_a%g' % a, pb, yt)
            add('between_a%g+time' % a, pb + tdev, yt)
        nrow += int(m.sum())
    table = {k: float(np.sqrt(v / nrow)) for k, v in sse.items()}
    gmae = {k: float(np.mean(v)) for k, v in gerr.items()}
    best = min(table, key=lambda k: (table[k], k))
    mu = P.mean(0)
    sd = P.std(0)
    sd[sd < 1e-12] = 1.0
    dsum = float(dens.sum())
    slope = float(nums.sum() / dsum) if dsum > 0 else 0.0
    tbar = float(X[:, it].mean())
    use_time = best.endswith('+time')
    coef = None
    alpha_used = None
    if best.startswith('between'):
        alpha_used = float(best.split('_a')[1].replace('+time', ''))
        w, b = _ridge((P - mu) / sd, Py, alpha_used)
        pred = ((Xet[:, bcols] - mu) / sd) @ w + b
        coef = {str(names[j]): round(float(w[k]), 4) for k, j in enumerate(bcols)}
    else:
        pred = np.full(len(Xe), med)
    if use_time:
        pred = pred + slope * (Xe[:, it] - tbar)
    lo, hi = float(y.min()), float(y.max())
    pred = np.clip(np.nan_to_num(pred, nan=med, posinf=hi, neginf=lo), lo, hi)
    findings = {
        'fallback_used': False,
        'feature_names': [str(n) for n in names],
        'roles': {'age': str(names[ia]), 'sex': str(names[isx]), 'time': str(names[it])},
        'log_transformed': [str(names[j]) for j in logcols],
        'n_train_patients': int(n_p),
        'n_train_rows': int(len(y)),
        'train_median': round(med, 4),
        'within_time_slope_per_day': round(slope, 6),
        'train_time_mean': round(tbar, 3),
        'chosen': best,
        'alpha_used': alpha_used,
        'time_component_used': bool(use_time),
        'lopo_row_rmse': {k: round(v, 4) for k, v in table.items()},
        'lopo_equal_patient_mae': {k: round(v, 4) for k, v in gmae.items()},
        'between_coef_standardized_patient_means': coef,
        'transductive_use_of_inputs': 'none; scaling and log clipping use train only',
        'prediction_clip_range': [round(lo, 3), round(hi, 3)]
    }
    return {'prediction': [float(v) for v in pred], 'findings': findings}
