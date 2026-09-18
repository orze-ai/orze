import numpy as np


def _ridge_fit(P, ybar, alpha):
    mu = P.mean(axis=0)
    sd = P.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (P - mu) / sd
    c = float(ybar.mean())
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (ybar - c))
    return (mu, sd, c, w)


def _ridge_pred(m, Q):
    mu, sd, c, w = m
    return c + ((Q - mu) / sd) @ w


def _run(train, inputs, seed):
    fn = [str(v) for v in train['feature_names']]
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    g = [str(v) for v in train['groups']]
    fne = [str(v) for v in inputs['feature_names']]
    Xe = np.asarray(inputs['X'], dtype=float)
    Xe = Xe[:, [fne.index(nm) for nm in fn]]
    i_age = fn.index('age')
    i_sex = fn.index('sex')
    i_tt = fn.index('test_time')
    wanted = ['age', 'sex', 'DFA', 'HNR', 'NHR', 'PPE', 'Shimmer(dB)']
    sets = {
        'all_no_time': [j for j in range(len(fn)) if j != i_tt],
        'reliable': [fn.index(nm) for nm in wanted if nm in fn],
        'age_sex': [i_age, i_sex],
    }
    pats = sorted(set(g))
    pidx = {p: np.asarray([i for i, v in enumerate(g) if v == p], dtype=int) for p in pats}
    npat = len(pats)
    alphas = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    weights = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0]
    tt = X[:, i_tt]
    res = {}
    full = {}
    for sname, keep in sets.items():
        P = np.asarray([X[pidx[p]][:, keep].mean(axis=0) for p in pats])
        yb = np.asarray([y[pidx[p]].mean() for p in pats])
        for a in alphas:
            pr = np.zeros(npat)
            for k in range(npat):
                m = np.ones(npat, dtype=bool)
                m[k] = False
                pr[k] = _ridge_pred(_ridge_fit(P[m], yb[m], a), P[k:k + 1])[0]
            res[(sname, a)] = pr
            full[(sname, a)] = _ridge_fit(P, yb, a)
    const = np.zeros(npat)
    slope = np.zeros(npat)
    for k, p in enumerate(pats):
        keepmask = np.asarray([q != p for q in g])
        const[k] = float(np.median(y[keepmask]))
        num = 0.0
        den = 0.0
        for q in pats:
            if q == p:
                continue
            ii = pidx[q]
            tc = tt[ii] - tt[ii].mean()
            yc = y[ii] - y[ii].mean()
            num += float(tc @ yc)
            den += float(tc @ tc)
        slope[k] = num / den if den > 1e-12 else 0.0

    def score(sname, a, w, use_slope):
        pr = res[(sname, a)]
        se = 0.0
        nr = 0
        pm = []
        for k, p in enumerate(pats):
            ii = pidx[p]
            base = w * pr[k] + (1.0 - w) * const[k]
            t = tt[ii]
            pred = base + (slope[k] * (t - t.mean()) if use_slope else 0.0)
            e = pred - y[ii]
            se += float(e @ e)
            nr += len(ii)
            pm.append(float(np.mean(np.abs(e))))
        return (se / nr) ** 0.5, float(np.mean(pm))

    table = []
    for sname in sets:
        for a in alphas:
            for w in weights:
                for us in (False, True):
                    r, m = score(sname, a, w, us)
                    table.append([round(r, 4), round(m, 4), sname, a, w, bool(us)])
    table.sort(key=lambda z: (z[0], z[4]))
    best = table[0]
    c_rmse, c_mae = score('age_sex', alphas[0], 0.0, False)
    cs_rmse, cs_mae = score('age_sex', alphas[0], 0.0, True)
    bs = best[2]
    ba = best[3]
    bw = best[4]
    bus = best[5]
    num = 0.0
    den = 0.0
    for p in pats:
        ii = pidx[p]
        tc = tt[ii] - tt[ii].mean()
        yc = y[ii] - y[ii].mean()
        num += float(tc @ yc)
        den += float(tc @ tc)
    slope_full = num / den if den > 1e-12 else 0.0
    const_full = float(np.median(y))
    mod = full[(bs, ba)]
    keep = sets[bs]
    gr = {}
    for i in range(Xe.shape[0]):
        kk = (round(float(Xe[i, i_age]), 4), round(float(Xe[i, i_sex]), 4))
        gr.setdefault(kk, []).append(i)
    pred = np.zeros(Xe.shape[0])
    gsizes = []
    gvals = []
    for kk in gr:
        rows_a = np.asarray(gr[kk], dtype=int)
        q = Xe[rows_a][:, keep].mean(axis=0).reshape(1, -1)
        rp = float(_ridge_pred(mod, q)[0])
        base = bw * rp + (1.0 - bw) * const_full
        te = Xe[rows_a, i_tt]
        pred[rows_a] = base + (slope_full * (te - te.mean()) if bus else 0.0)
        gsizes.append(len(rows_a))
        gvals.append(round(base, 3))
    pred = np.clip(pred, 0.0, 60.0)
    findings = {
        'design': 'pseudo_patient_grouping_by_exact_age_sex_then_LOPO_selected_patient_level_shrunk_ridge_plus_optional_time_slope',
        'fallback_used': False,
        'chosen': {'featset': bs, 'alpha': ba, 'weight': bw, 'use_time_slope': bus},
        'lopo_row_rmse_best': best[0],
        'lopo_group_mae_best': best[1],
        'lopo_constant_only_rmse_gmae': [round(c_rmse, 4), round(c_mae, 4)],
        'lopo_constant_plus_slope_rmse_gmae': [round(cs_rmse, 4), round(cs_mae, 4)],
        'lopo_top10_rmse_gmae_set_alpha_weight_slope': table[:10],
        'within_patient_time_slope_per_day': round(slope_full, 5),
        'n_train_patients': npat,
        'n_pseudo_groups_eval': len(gr),
        'pseudo_group_sizes_desc': sorted(gsizes, reverse=True)[:25],
        'pseudo_group_base_predictions_sorted': sorted(gvals)[:25],
        'pred_summary': {'n': int(Xe.shape[0]), 'mean': round(float(pred.mean()), 3), 'sd': round(float(pred.std()), 3), 'min': round(float(pred.min()), 3), 'max': round(float(pred.max()), 3)},
        'train_median_constant': round(const_full, 3),
        'transductive_note': 'unlabeled evaluation X used only to group rows by exact (age,sex) and average features within those pseudo-patient groups; no evaluation labels used',
    }
    return {'prediction': [float(v) for v in pred], 'findings': findings}


def fit_predict(train, inputs, seed):
    try:
        return _run(train, inputs, seed)
    except Exception as exc:
        med = float(np.median(np.asarray(train['y'], dtype=float)))
        n = len(inputs['X'])
        return {'prediction': [med] * n,
                'findings': {'fallback_used': True,
                             'fallback_reason': str(exc)[:300],
                             'design': 'pseudo_patient_grouping_by_exact_age_sex_then_LOPO_selected_patient_level_shrunk_ridge_plus_optional_time_slope'}}
