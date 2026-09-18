import numpy as np


def analyze(data, history, seed):
    try:
        return _run(data, history, seed)
    except Exception as e:
        return {'fallback_used': True, 'error': str(e)[:500]}


def _run(data, history, seed):
    rng = np.random.default_rng(int(seed) if seed is not None else 0)
    tr = data['train']
    dv = data['development']
    fn = list(tr['feature_names'])
    X = np.array(list(tr['X']) + list(dv['X']), dtype=float)
    y = np.array(list(tr['y']) + list(dv['y']), dtype=float)
    g = np.array(['t%s' % str(v) for v in tr['groups']] + ['d%s' % str(v) for v in dv['groups']])
    N = len(y)
    voice = [i for i, n in enumerate(fn) if n not in ('age', 'sex', 'test_time')]
    Z = X.copy()
    logged = []
    for i in voice:
        if np.min(X[:, i]) > 0:
            Z[:, i] = np.log(X[:, i])
            logged.append(fn[i])
    keep = [i for i, n in enumerate(fn) if n != 'test_time']
    keep_names = [fn[i] for i in keep]
    allc = list(range(len(keep)))
    vkeep = [j for j, i in enumerate(keep) if fn[i] not in ('age', 'sex')]
    askeep = [j for j, i in enumerate(keep) if fn[i] in ('age', 'sex')]
    pats = sorted(set(g.tolist()))
    P = len(pats)
    idx = [np.where(g == p)[0] for p in pats]
    PX = np.array([Z[ix][:, keep].mean(axis=0) for ix in idx])
    PY = np.array([y[ix].mean() for ix in idx])
    PN = np.array([len(ix) for ix in idx], dtype=float)
    PD = np.array([p[0] == 'd' for p in pats])
    gm = float(y.mean())
    between = float(np.sum(PN * (PY - gm) ** 2) / N)
    within = float(sum(np.sum((y[ix] - PY[k]) ** 2) for k, ix in enumerate(idx)) / N)

    def std_fit(A0, B0):
        mu = A0.mean(0)
        sd = A0.std(0)
        sd[sd == 0] = 1.0
        return (A0 - mu) / sd, (B0 - mu) / sd

    def ridge(Xa, ya, Xb, alpha, cols):
        A, B = std_fit(Xa[:, cols], Xb[:, cols])
        ym = ya.mean()
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ya - ym))
        return B @ w + ym

    def knn(Xa, ya, Xb, k, cols):
        A, B = std_fit(Xa[:, cols], Xb[:, cols])
        out = []
        for b in B:
            d = np.sqrt(((A - b) ** 2).sum(1))
            o = np.argsort(d)[:k]
            out.append(ya[o].mean())
        return np.array(out)

    def m_median(ti, si):
        rows = np.concatenate([idx[k] for k in ti])
        return np.full(len(si), np.median(y[rows]))

    def m_pmean(ti, si):
        return np.full(len(si), PY[ti].mean())

    def mk_ridge(alpha, cols):
        return lambda ti, si: ridge(PX[ti], PY[ti], PX[si], alpha, cols)

    def mk_knn(k):
        return lambda ti, si: knn(PX[ti], PY[ti], PX[si], k, allc)

    def mk_blend(lam, alpha):
        r = mk_ridge(alpha, allc)
        return lambda ti, si: lam * r(ti, si) + (1.0 - lam) * m_median(ti, si)

    models = {'median': m_median, 'pmean': m_pmean}
    for a in (1, 10, 100, 1000, 10000):
        models['ridge_a%g' % a] = mk_ridge(a, allc)
    models['ridge_voice_a100'] = mk_ridge(100, vkeep)
    models['ridge_voice_a1000'] = mk_ridge(1000, vkeep)
    models['ridge_agesex_a1'] = mk_ridge(1, askeep)
    models['knn5'] = mk_knn(5)
    models['knn9'] = mk_knn(9)
    for lam in (0.25, 0.5, 0.75):
        models['blend%.2f_a1000' % lam] = mk_blend(lam, 1000)

    def lopo(f, pool):
        pred = np.full(N, np.nan)
        for k in pool:
            ti = np.array([j for j in pool if j != k])
            pred[idx[k]] = f(ti, np.array([k]))[0]
        rows = np.concatenate([idx[k] for k in pool])
        rmse = float(np.sqrt(np.mean((pred[rows] - y[rows]) ** 2)))
        pmae = np.array([np.mean(np.abs(pred[idx[k]] - y[idx[k]])) for k in pool])
        signed = np.array([pred[idx[k]][0] - PY[k] for k in pool])
        return rmse, pmae, signed

    pool35 = list(range(P))
    base_rmse, base_pmae, base_signed = lopo(m_median, pool35)
    lopo35 = {}
    for name, f in models.items():
        rmse, pmae, signed = lopo(f, pool35)
        worst = np.argsort(-np.abs(signed))[:2]
        lopo35[name] = {'rmse': round(rmse, 3), 'mean_pmae': round(float(pmae.mean()), 3),
                        'wins_vs_median': int(np.sum(pmae < base_pmae - 1e-9)), 'of': P,
                        'worst2_signed': [[pats[pool35[w]], round(float(signed[w]), 2)] for w in worst]}

    def lopo_ridge_rmse(PYv, alpha):
        se = 0.0
        for k in range(P):
            ti = np.array([j for j in range(P) if j != k])
            c = ridge(PX[ti], PYv[ti], PX[[k]], alpha, allc)[0]
            se += np.sum((c - (y[idx[k]] - PY[k] + PYv[k])) ** 2)
        return float(np.sqrt(se / N))

    obs = lopo_ridge_rmse(PY, 1000)
    nperm = 300
    perm = np.array([lopo_ridge_rmse(rng.permutation(PY), 1000) for _ in range(nperm)])
    pval = float((np.sum(perm <= obs) + 1) / (nperm + 1))

    ti = np.array([k for k in range(P) if not PD[k]])
    si = np.array([k for k in range(P) if PD[k]])
    dev_rows = np.concatenate([idx[k] for k in si])
    proto = {}
    for name in ('median', 'pmean', 'ridge_a100', 'ridge_a1000', 'ridge_a10000', 'ridge_voice_a1000', 'ridge_agesex_a1', 'knn5', 'blend0.25_a1000', 'blend0.50_a1000'):
        c = models[name](ti, si)
        pred = np.full(N, np.nan)
        for j, k in enumerate(si):
            pred[idx[k]] = c[j]
        rmse = float(np.sqrt(np.mean((pred[dev_rows] - y[dev_rows]) ** 2)))
        pmae = [float(np.mean(np.abs(pred[idx[k]] - y[idx[k]]))) for k in si]
        proto[name] = {'dev_rmse': round(rmse, 3), 'dev_mean_pmae': round(float(np.mean(pmae)), 3),
                       'signed_by_dev_patient': [round(float(c[j] - PY[k]), 2) for j, k in enumerate(si)]}
    train_rows = np.concatenate([idx[k] for k in ti])
    tmed = float(np.median(y[train_rows]))

    def rank(a):
        r = np.empty(len(a))
        r[np.argsort(a)] = np.arange(len(a))
        return r
    sp = {keep_names[j]: round(float(np.corrcoef(rank(PX[:, j]), rank(PY))[0, 1]), 3) for j in range(len(keep))}

    return {
        'fallback_used': False,
        'n_patients': P, 'n_train_patients': int((~PD).sum()), 'n_dev_patients': int(PD.sum()), 'n_rows': N,
        'history_len': len(history), 'logged_features': logged,
        'y_var_total': round(float(y.var()), 3), 'y_var_between_patient': round(between, 3), 'y_var_within_patient': round(within, 3),
        'between_fraction': round(between / max(y.var(), 1e-12), 3),
        'train_median': round(tmed, 3),
        'dev_patients': [{'id': pats[k], 'n': int(PN[k]), 'mean_y': round(float(PY[k]), 2), 'offset_vs_train_median': round(float(PY[k] - tmed), 2)} for k in si],
        'lopo35': lopo35,
        'permutation_ridge_a1000': {'observed_rmse': round(obs, 3), 'null_mean': round(float(perm.mean()), 3), 'null_min': round(float(perm.min()), 3), 'null_5pct': round(float(np.percentile(perm, 5)), 3), 'p_value': round(pval, 4), 'n_perm': nperm},
        'protocol_28_to_7': proto,
        'spearman_patient_mean_feature_vs_mean_updrs': sp,
    }
