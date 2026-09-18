import numpy as np
import json
import math

def analyze(data, history, seed):
    out = {'fallback_used': False}
    try:
        try:
            s = int(seed)
        except Exception:
            s = 0
        rng = np.random.default_rng(20260918 + s)
        r = lambda v: float(round(float(v), 3))
        r4 = lambda v: float(round(float(v), 4))
        fn = [str(v) for v in data['train']['feature_names']]
        Xs = []
        ys = []
        gs = []
        for key, pre in (('train', 't'), ('development', 'd')):
            d = data[key]
            for xi, yi, gi in zip(d['X'], d['y'], d['groups']):
                Xs.append([float(v) for v in xi])
                ys.append(float(yi))
                gs.append(pre + str(gi))
        X = np.array(Xs, dtype=float)
        y = np.array(ys, dtype=float)
        pats = sorted(set(gs))
        pidx = {p: i for i, p in enumerate(pats)}
        gidx = np.array([pidx[v] for v in gs])
        npat = len(pats)
        nrow = len(y)
        voice_cols = [i for i, n in enumerate(fn) if n not in ('age', 'sex', 'test_time')]
        tcol = fn.index('test_time') if 'test_time' in fn else None
        Xl = X.copy()
        logged = []
        for i in voice_cols:
            if np.all(X[:, i] > 0):
                Xl[:, i] = np.log(X[:, i])
                logged.append(fn[i])
        P = np.zeros((npat, X.shape[1]))
        pm = np.zeros(npat)
        n_p = np.zeros(npat, dtype=int)
        for k in range(npat):
            m = gidx == k
            P[k] = Xl[m].mean(axis=0)
            pm[k] = y[m].mean()
            n_p[k] = int(m.sum())
        is_dev = np.array([p.startswith('d') for p in pats])
        dev_rows = np.array([v.startswith('d') for v in gs])

        def ridge(Ptr, ytr, Pte, alpha):
            mu = Ptr.mean(axis=0)
            sd = Ptr.std(axis=0)
            sd[sd == 0] = 1.0
            Z = (Ptr - mu) / sd
            ym = ytr.mean()
            w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (ytr - ym))
            return ((Pte - mu) / sd) @ w + ym

        def lopo_ridge(cols, alpha, target):
            ppred = np.zeros(npat)
            for k in range(npat):
                tr = np.ones(npat, dtype=bool)
                tr[k] = False
                ppred[k] = ridge(P[tr][:, cols], target[tr], P[k:k + 1][:, cols], alpha)[0]
            return ppred

        def lopo_median(yy):
            ppred = np.zeros(npat)
            for k in range(npat):
                ppred[k] = np.median(yy[gidx != k])
            return ppred

        def row_sse(ppred, yy):
            res = ppred[gidx] - yy
            sse = np.zeros(npat)
            for k in range(npat):
                sse[k] = float(np.sum(res[gidx == k] ** 2))
            return sse

        def rmse_from(ppred, yy):
            return float(np.sqrt(np.mean((ppred[gidx] - yy) ** 2)))

        def pmae_from(ppred, yy):
            return float(np.mean([np.mean(np.abs(ppred[k] - yy[gidx == k])) for k in range(npat)]))

        sets = {
            'all': [i for i, n in enumerate(fn) if n != 'test_time'],
            'voice': voice_cols,
            'age': [i for i, n in enumerate(fn) if n == 'age'],
            'agesex': [i for i, n in enumerate(fn) if n in ('age', 'sex')],
            'jitter': [i for i, n in enumerate(fn) if 'Jitter' in n],
            'shimmer': [i for i, n in enumerate(fn) if 'Shimmer' in n],
            'noise': [i for i, n in enumerate(fn) if n in ('NHR', 'HNR')],
            'nonlinear': [i for i, n in enumerate(fn) if n in ('RPDE', 'DFA', 'PPE')],
        }
        sets = {k: v for k, v in sets.items() if len(v) > 0}
        med_p = lopo_median(y)
        med_rmse = rmse_from(med_p, y)
        med_sse = row_sse(med_p, y)
        worst = int(np.argmax(np.abs(med_p - pm)))
        keep = np.ones(npat, dtype=bool)
        keep[worst] = False

        def rmse_excl(sse):
            return float(np.sqrt(sse[keep].sum() / n_p[keep].sum()))

        out['n_patients'] = int(npat)
        out['n_rows'] = int(nrow)
        out['n_dev_patients'] = int(is_dev.sum())
        out['logged_features'] = logged
        out['lopo35_median'] = {
            'rmse': r(med_rmse),
            'pmae': r(pmae_from(med_p, y)),
            'rmse_excl_worst': r(rmse_excl(med_sse)),
            'worst_patient': pats[worst],
            'dev7_rmse_lopo': r(np.sqrt(med_sse[is_dev].sum() / n_p[is_dev].sum())),
            'train28_rmse_lopo': r(np.sqrt(med_sse[~is_dev].sum() / n_p[~is_dev].sum())),
        }
        abl = {}
        for name, cols in sets.items():
            alphas = (100.0, 1000.0, 10000.0) if name in ('all', 'voice') else (1000.0,)
            for alpha in alphas:
                pp = lopo_ridge(cols, alpha, pm)
                sse = row_sse(pp, y)
                abl[name + '_a' + str(int(alpha))] = {
                    'rmse': r(np.sqrt(sse.sum() / nrow)),
                    'pmae': r(pmae_from(pp, y)),
                    'rmse_excl_worst': r(rmse_excl(sse)),
                    'wins_vs_median': int(np.sum(sse < med_sse)),
                    'dev7_rmse_lopo': r(np.sqrt(sse[is_dev].sum() / n_p[is_dev].sum())),
                    'train28_rmse_lopo': r(np.sqrt(sse[~is_dev].sum() / n_p[~is_dev].sum())),
                }
        out['lopo35_ridge_ablation'] = abl

        ref = lopo_ridge(sets['all'], 1000.0, pm)
        ref_sse = row_sse(ref, y)
        obs_ridge = float(np.sqrt(ref_sse.sum() / nrow))
        obs_diff = med_rmse - obs_ridge
        B = 2000
        diffs = np.zeros(B)
        for b in range(B):
            idx = rng.integers(0, npat, npat)
            diffs[b] = np.sqrt(med_sse[idx].sum() / n_p[idx].sum()) - np.sqrt(ref_sse[idx].sum() / n_p[idx].sum())
        out['patient_bootstrap_median_minus_ridge_all_a1000'] = {
            'n_boot': int(B),
            'observed': r(obs_diff),
            'ci95': [r(np.percentile(diffs, 2.5)), r(np.percentile(diffs, 97.5))],
            'frac_ridge_better': r(np.mean(diffs > 0)),
            'note': 'approximate: patients resampled from fixed LOPO residuals, fits not re-run',
        }

        NP = 1000
        null_diff = np.zeros(NP)
        null_ridge = np.zeros(NP)
        for i in range(NP):
            perm = rng.permutation(npat)
            pm_perm = pm[perm]
            yp = y - pm[gidx] + pm_perm[gidx]
            rp = lopo_ridge(sets['all'], 1000.0, pm_perm)
            mp = lopo_median(yp)
            rr = rmse_from(rp, yp)
            mr = rmse_from(mp, yp)
            null_ridge[i] = rr
            null_diff[i] = mr - rr
        out['permutation_all_a1000'] = {
            'n_perm': int(NP),
            'observed_ridge_rmse': r(obs_ridge),
            'observed_median_minus_ridge': r(obs_diff),
            'p_ridge_rmse': r((np.sum(null_ridge <= obs_ridge) + 1) / (NP + 1)),
            'p_diff': r((np.sum(null_diff >= obs_diff) + 1) / (NP + 1)),
            'null_diff_mean': r(null_diff.mean()),
            'null_diff_95pct': r(np.percentile(null_diff, 95)),
            'null_ridge_5pct': r(np.percentile(null_ridge, 5)),
            'note': 'patient means permuted across patients, within-patient deviations preserved',
        }

        tr = ~is_dev
        p28 = ridge(P[tr][:, sets['all']], pm[tr], P[is_dev][:, sets['all']], 1000.0)
        pp = np.zeros(npat)
        pp[is_dev] = p28
        sse28 = row_sse(pp, y)
        med28 = float(np.median(y[~dev_rows]))
        out['protocol_28_to_7'] = {
            'median_dev_rmse': r(np.sqrt(np.mean((med28 - y[dev_rows]) ** 2))),
            'ridge_all_a1000_dev_rmse': r(np.sqrt(sse28[is_dev].sum() / n_p[is_dev].sum())),
            'train_median': r(med28),
        }

        if tcol is not None:
            t = X[:, tcol]
            devy = y - pm[gidx]
            tm = np.array([t[gidx == k].mean() for k in range(npat)])
            dt = t - tm[gidx]
            slopes = np.zeros(npat)
            for k in range(npat):
                m = gidx == k
                den = float(np.sum(dt[m] ** 2))
                slopes[k] = float(np.sum(dt[m] * devy[m]) / den) if den > 0 else 0.0
            pooled = float(np.sum(dt * devy) / np.sum(dt ** 2))
            within_var = float(np.mean(devy ** 2))
            pred_dev = np.zeros(nrow)
            for k in range(npat):
                m = gidx == k
                o = ~m
                sl = float(np.sum(dt[o] * devy[o]) / np.sum(dt[o] ** 2))
                pred_dev[m] = sl * dt[m]
            resid = devy - pred_dev
            out['within_patient_time_trend'] = {
                'within_var': r(within_var),
                'total_var': r(np.var(y)),
                'between_var': r(np.var(pm[gidx])),
                'pooled_slope_updrs_per_day': r4(pooled),
                'per_patient_slope_median': r4(np.median(slopes)),
                'per_patient_slope_q25_q75': [r4(np.percentile(slopes, 25)), r4(np.percentile(slopes, 75))],
                'frac_patient_slope_positive': r(np.mean(slopes > 0)),
                'lopo_r2_within': r(1 - np.mean(resid ** 2) / within_var),
                'oracle_mean_rmse': r(np.sqrt(within_var)),
                'oracle_mean_plus_lopo_slope_rmse': r(np.sqrt(np.mean(resid ** 2))),
                'note': 'oracle patient mean is unavailable to fit_predict; bounds what test_time adds beyond an unknown patient offset',
            }

        out['interpretation_guide'] = 'H1 (real between-patient signal): bootstrap ci95 excludes 0, p_diff<0.05, voice families beat median, effect survives excl_worst. H2 (no detectable transfer): ci95 covers 0, p_diff>0.1, or effect driven by age/worst patient. Selection this round is unaffected: lowest valid dev RMSE remains the train median.'
        while len(json.dumps(out).encode('utf-8')) > 8000 and len(abl) > 0:
            abl.pop(list(abl.keys())[-1])
        return out
    except Exception as e:
        return {'fallback_used': True, 'error': str(e)[:400]}
