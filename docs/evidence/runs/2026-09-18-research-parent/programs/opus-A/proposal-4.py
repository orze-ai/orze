import json, math
import numpy as np

EPS = 1e-9
SETS = [('A',), ('A', 'M'), ('A', 'D'), ('A', 'M', 'D'), ('A', 'M', 'S'), ('A', 'M', 'D', 'S'), ('M',), ('A', 'M', 'D', 'S', 'N')]
ALPHAS = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
TMULTS = [0.0, 1.0]
NPERM = 100


def _locate(names):
    low = [str(n).strip().lower() for n in names]
    def f(k):
        return low.index(k) if k in low else -1
    ai = f('age')
    si = f('sex')
    ti = f('test_time')
    if ti < 0:
        ti = f('testtime')
    vc = [i for i in range(len(low)) if i not in (ai, si, ti)]
    return ai, si, ti, vc


def _blocks(Xg, ai, si, ti, vc):
    t = Xg[:, ti].astype(float)
    tc = t - t.mean()
    den = float(np.dot(tc, tc))
    span = float(t.max() - t.min())
    nv = len(vc)
    med = np.zeros(nv)
    dsp = np.zeros(nv)
    drf = np.zeros(nv)
    for k, j in enumerate(vc):
        v = Xg[:, j].astype(float)
        m = float(np.median(v))
        q1, q3 = np.percentile(v, [25.0, 75.0])
        sc = abs(m) + EPS
        med[k] = m
        dsp[k] = math.log1p(max(0.0, float(q3 - q1)) / sc)
        sl = float(np.dot(tc, v - v.mean()) / den) if den > 1e-12 else 0.0
        drf[k] = sl * span / sc
    A = np.array([float(Xg[0, ai]) if ai >= 0 else 0.0, float(Xg[0, si]) if si >= 0 else 0.0])
    N = np.array([span, math.log(float(Xg.shape[0]) + 1.0)])
    out = {'A': A, 'M': med, 'D': dsp, 'S': drf, 'N': N}
    bad = 0
    for k in list(out.keys()):
        arr = np.asarray(out[k], dtype=float)
        bad += int(np.sum(~np.isfinite(arr)))
        out[k] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return out, float(t.mean()), bad


def _design(bl, sets):
    return np.array([np.concatenate([b[s] for s in sets]) for b in bl], dtype=float)


def _opmat(Z, alpha, n):
    M = np.zeros((n, n))
    eye = None
    for h in range(n):
        tr = [i for i in range(n) if i != h]
        A = Z[tr]
        mu = A.mean(0)
        sd = A.std(0)
        sd[sd < 1e-12] = 1.0
        An = (A - mu) / sd
        Bn = (Z[h:h + 1] - mu) / sd
        if eye is None or eye.shape[0] != An.shape[1]:
            eye = np.eye(An.shape[1])
        G = An.T @ An + alpha * eye
        u = (Bn @ np.linalg.solve(G, An.T))[0]
        M[h, tr] = 1.0 / (n - 1) + u - float(u.mean())
    return M


def _fit_full(Z, y, alpha):
    mu = Z.mean(0)
    sd = Z.std(0)
    sd[sd < 1e-12] = 1.0
    A = (Z - mu) / sd
    ym = float(y.mean())
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y - ym))
    return mu, sd, ym, w


def _sse(c, d, cnt, Sy, Syy, Sdd, Syd):
    return float(np.sum(cnt * c * c + d * d * Sdd + Syy - 2.0 * c * Sy - 2.0 * d * Syd))


def _run(train, inputs, seed, findings):
    Xtr = np.nan_to_num(np.asarray(train['X'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ytr = np.asarray(train['y'], dtype=float)
    groups = [str(g) for g in train['groups']]
    fn = list(train['feature_names'])
    ai, si, ti, vc = _locate(fn)
    idx = {}
    order = []
    for i, g in enumerate(groups):
        if g not in idx:
            idx[g] = []
            order.append(g)
        idx[g].append(i)
    pats = []
    nbad = 0
    for g in order:
        rows = np.asarray(idx[g], dtype=int)
        b, tm, bad = _blocks(Xtr[rows], ai, si, ti, vc)
        nbad += bad
        pats.append({'rows': rows, 'b': b, 'tm': tm})
    n = len(pats)
    ptar = np.array([float(ytr[p['rows']].mean()) for p in pats])
    cnt = np.array([float(len(p['rows'])) for p in pats])
    Sy = np.zeros(n)
    Syy = np.zeros(n)
    Sdd = np.zeros(n)
    Syd = np.zeros(n)
    for h, p in enumerate(pats):
        yy = ytr[p['rows']]
        dt = Xtr[p['rows'], ti] - p['tm']
        Sy[h] = float(yy.sum())
        Syy[h] = float(np.dot(yy, yy))
        Sdd[h] = float(np.dot(dt, dt))
        Syd[h] = float(np.dot(yy, dt))
    total_rows = float(cnt.sum())
    Zall = {s: _design([p['b'] for p in pats], s) for s in SETS}
    slope_full = float(np.sum(Syd) / max(float(np.sum(Sdd)), 1e-12))
    Mc = np.full((n, n), 1.0 / (n - 1))
    np.fill_diagonal(Mc, 0.0)
    gm = Mc @ ptar
    opm = {}
    for s in SETS:
        for a in ALPHAS:
            opm[(s, a)] = _opmat(Zall[s], a, n)
    tot_syd = float(np.sum(Syd))
    tot_sdd = float(np.sum(Sdd))
    slopes_lopo = np.array([(tot_syd - Syd[h]) / max(tot_sdd - Sdd[h], 1e-12) for h in range(n)])

    def rowrmse(c, d):
        return math.sqrt(_sse(c, d, cnt, Sy, Syy, Sdd, Syd) / total_rows)

    res = []
    for key, M in opm.items():
        o = M @ ptar
        for lam in LAMS:
            c = gm + lam * (o - gm)
            pr = math.sqrt(float(np.mean((c - ptar) ** 2)))
            for tmv in TMULTS:
                d = tmv * slopes_lopo
                res.append({'set': '+'.join(key[0]), 'alpha': key[1], 'lam': lam, 'tmult': tmv,
                            'lopo_row_rmse': rowrmse(c, d), 'lopo_patient_rmse': pr, 'k': key})
    res.sort(key=lambda z: z['lopo_row_rmse'])
    best = res[0]
    zero = np.zeros(n)
    const_row = rowrmse(gm, zero)
    const_row_time = rowrmse(gm, slopes_lopo)
    const_pat = math.sqrt(float(np.mean((gm - ptar) ** 2)))
    medc = np.array([float(np.median(np.delete(ytr, pats[h]['rows']))) for h in range(n)])
    const_rowmed = rowrmse(medc, zero)

    nested_sse = 0.0
    choice = {}
    for h in range(n):
        tr = [i for i in range(n) if i != h]
        m_in = n - 1
        y_in = ptar[tr]
        cnt_i = cnt[tr]
        Sy_i = Sy[tr]
        Syy_i = Syy[tr]
        Sdd_i = Sdd[tr]
        Syd_i = Syd[tr]
        tot_i = float(cnt_i.sum())
        Mc_i = np.full((m_in, m_in), 1.0 / (m_in - 1))
        np.fill_diagonal(Mc_i, 0.0)
        gm_i = Mc_i @ y_in
        s_syd = float(np.sum(Syd_i))
        s_sdd = float(np.sum(Sdd_i))
        sl_i = np.array([(s_syd - Syd_i[q]) / max(s_sdd - Sdd_i[q], 1e-12) for q in range(m_in)])
        bi = None
        for s in SETS:
            Zs = Zall[s][tr]
            for a in ALPHAS:
                o = _opmat(Zs, a, m_in) @ y_in
                for lam in LAMS:
                    c = gm_i + lam * (o - gm_i)
                    for tmv in TMULTS:
                        rr = math.sqrt(_sse(c, tmv * sl_i, cnt_i, Sy_i, Syy_i, Sdd_i, Syd_i) / tot_i)
                        if bi is None or rr < bi[0]:
                            bi = (rr, s, a, lam, tmv)
        s, a, lam, tmv = bi[1], bi[2], bi[3], bi[4]
        mu, sd, ym, w = _fit_full(Zall[s][tr], y_in, a)
        base_h = float(ym + (((Zall[s][h:h + 1] - mu) / sd) @ w)[0])
        gh = float(y_in.mean())
        ch = np.array([gh + lam * (base_h - gh)])
        dh = np.array([tmv * ((tot_syd - Syd[h]) / max(tot_sdd - Sdd[h], 1e-12))])
        nested_sse += _sse(ch, dh, cnt[h:h + 1], Sy[h:h + 1], Syy[h:h + 1], Sdd[h:h + 1], Syd[h:h + 1])
        kk = '+'.join(s) + '|a' + str(a) + '|l' + str(lam) + '|t' + str(tmv)
        choice[kk] = choice.get(kk, 0) + 1
    nested_row_rmse = math.sqrt(nested_sse / total_rows)

    rng = np.random.default_rng(int(seed) if seed is not None else 0)
    obs_best_pat = min(r['lopo_patient_rmse'] for r in res)
    obs_gain = const_pat - obs_best_pat
    ge = 0
    for _ in range(NPERM):
        yp = ptar[rng.permutation(n)]
        gmp = Mc @ yp
        cp = math.sqrt(float(np.mean((gmp - yp) ** 2)))
        bp = None
        for key, M in opm.items():
            o = M @ yp
            for lam in LAMS:
                c = gmp + lam * (o - gmp)
                r = math.sqrt(float(np.mean((c - yp) ** 2)))
                if bp is None or r < bp:
                    bp = r
        if (cp - bp) >= obs_gain - 1e-12:
            ge += 1
    perm_p = (1.0 + ge) / (NPERM + 1.0)

    Xe = np.nan_to_num(np.asarray(inputs['X'], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    fne = [str(x).strip().lower() for x in inputs['feature_names']]
    low_tr = [str(x).strip().lower() for x in fn]
    cols = []
    mapped = True
    for nm in low_tr:
        if nm in fne:
            cols.append(fne.index(nm))
        else:
            mapped = False
            break
    if mapped and len(cols) == Xe.shape[1]:
        Xe = Xe[:, cols]
    else:
        mapped = False
    ne = int(Xe.shape[0])
    ymin = float(ytr.min())
    ymax = float(ytr.max())
    gmean_all = float(ptar.mean())
    clusters = {}
    gate_ok = False
    sizes = []
    if mapped and ai >= 0 and si >= 0:
        for i in range(ne):
            k = (round(float(Xe[i, ai]), 4), round(float(Xe[i, si]), 4))
            clusters.setdefault(k, []).append(i)
        sizes = [len(v) for v in clusters.values()]
        ntt = [len(set(np.round(Xe[np.asarray(v, dtype=int), ti], 4).tolist())) for v in clusters.values()]
        gate_ok = (2 <= len(clusters) <= 16) and min(sizes) >= 20 and min(ntt) >= 5
    pred = np.full(ne, gmean_all, dtype=float)
    disp_ratio = None
    if gate_ok:
        keys = list(clusters.keys())
        cb = []
        ctm = []
        for k in keys:
            rows = np.asarray(clusters[k], dtype=int)
            b, tmn, bad = _blocks(Xe[rows], ai, si, ti, vc)
            cb.append(b)
            ctm.append(tmn)
        s = best['k'][0]
        a = best['k'][1]
        lam = best['lam']
        tmv = best['tmult']
        mu, sd, ym, w = _fit_full(Zall[s], ptar, a)
        base = ym + ((_design(cb, s) - mu) / sd) @ w
        for ci, k in enumerate(keys):
            rows = np.asarray(clusters[k], dtype=int)
            c = gmean_all + lam * (float(base[ci]) - gmean_all)
            pred[rows] = c + tmv * slope_full * (Xe[rows, ti] - ctm[ci])
        dtr = np.median(np.array([p['b']['D'] for p in pats]), axis=0)
        dev = np.median(np.array([b['D'] for b in cb]), axis=0)
        disp_ratio = float(np.median((dev + 1e-9) / (dtr + 1e-9)))
    else:
        findings['fallback'] = {'used': True, 'reason': 'evaluation patient-cluster gate failed or feature names unmapped; predicted equal-patient train mean plus pooled time slope'}
        if mapped:
            tmed = float(np.median(Xe[:, ti]))
            pred = gmean_all + slope_full * (Xe[:, ti] - tmed)
        else:
            pred = np.full(ne, gmean_all, dtype=float)
    pred = np.clip(np.nan_to_num(np.asarray(pred, dtype=float), nan=gmean_all, posinf=ymax, neginf=ymin), ymin, ymax)

    top = []
    for r in res[:6]:
        top.append({'set': r['set'], 'alpha': r['alpha'], 'lam': r['lam'], 'tmult': r['tmult'],
                    'lopo_row_rmse': round(r['lopo_row_rmse'], 5), 'lopo_patient_rmse': round(r['lopo_patient_rmse'], 5)})
    ch = sorted(choice.items(), key=lambda z: -z[1])[:5]
    findings['blocks'] = {'A': 'age,sex', 'M': 'per-patient median of 16 voice measures', 'D': 'log1p of relative IQR (repeated-measurement dispersion)', 'S': 'relative temporal drift = OLS slope*span/|median|', 'N': 'observation span, log n_recordings'}
    findings['lopo'] = {'n_patients': n, 'n_rows': int(total_rows), 'n_nonfinite_blocked': int(nbad)}
    findings['reference_lopo'] = {'equal_patient_mean_row_rmse': round(const_row, 5),
                                  'equal_patient_mean_plus_time_row_rmse': round(const_row_time, 5),
                                  'row_median_constant_row_rmse': round(const_rowmed, 5),
                                  'equal_patient_mean_patient_rmse': round(const_pat, 5)}
    findings['selected'] = {'set': best['set'], 'alpha': best['alpha'], 'lam': best['lam'], 'tmult': best['tmult'],
                            'lopo_row_rmse': round(best['lopo_row_rmse'], 5),
                            'relative_gain_vs_constant_row': round((const_row - best['lopo_row_rmse']) / const_row, 5)}
    findings['top_combos'] = top
    findings['nested_lopo'] = {'row_rmse': round(nested_row_rmse, 5),
                               'relative_gain_vs_constant_row': round((const_row - nested_row_rmse) / const_row, 5),
                               'selection_inflation_row_rmse': round(nested_row_rmse - best['lopo_row_rmse'], 5),
                               'most_frequent_inner_choices': [{'cfg': k, 'folds': v} for k, v in ch]}
    findings['permutation_null'] = {'n_perm': NPERM, 'statistic': 'max over (set,alpha,lam) of patient-level LOPO RMSE gain vs equal-patient-mean constant',
                                    'observed_gain': round(obs_gain, 5), 'p_value_ge_observed': round(perm_p, 4)}
    findings['pooled_within_patient_time_slope'] = round(slope_full, 6)
    findings['eval'] = {'n_rows': ne, 'gate_ok': bool(gate_ok), 'n_clusters': len(clusters),
                        'min_size': int(min(sizes)) if sizes else 0, 'max_size': int(max(sizes)) if sizes else 0,
                        'dispersion_ratio_vs_train_patients': disp_ratio}
    findings['transductive_note'] = 'evaluation rows carry no group labels; rows are grouped by exact (age,sex) and patient-level medians/dispersion/drift are computed from unlabeled evaluation inputs only. No evaluation labels are used anywhere.'
    findings['caveats'] = 'Plain LOPO also chose hyperparameters, so it is optimistic; the nested estimate is the honest internal number. 28 training and 7 evaluation patients are a small group-level sample; no evaluation-label metric is computed here. Permutation null tests patient-level transfer only (tmult=0 axis of the statistic).'
    return {'prediction': [float(v) for v in pred], 'findings': findings}


def fit_predict(train, inputs, seed=0):
    findings = {'design': 'patient-level representation of repeated voice measurements (median location, log relative IQR dispersion, relative temporal drift) -> shrunken between-patient ridge with nested-LOPO honest estimate and patient-label permutation null; transductive (age,sex) recovery of evaluation patients',
                'fallback': {'used': False, 'reason': ''}}
    try:
        return _run(train, inputs, seed, findings)
    except Exception as exc:
        findings['fallback'] = {'used': True, 'reason': 'exception in main path: ' + type(exc).__name__}
        y = np.asarray(train['y'], dtype=float)
        v = float(np.median(y)) if y.size else 0.0
        m = len(inputs['X'])
        return {'prediction': [v] * m, 'findings': findings}
