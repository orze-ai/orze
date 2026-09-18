import numpy as np, json, math
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

def _r(x, d=3):
    try:
        x = float(x)
    except Exception:
        return None
    return round(x, d) if math.isfinite(x) else None

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return _r(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o

def analyze(data, history, seed):
    rng = np.random.RandomState(int(seed) if seed is not None else 0)
    fn = list(data['train']['feature_names'])
    idx = {n: i for i, n in enumerate(fn)}
    voice = [n for n in fn if n not in ('age', 'sex', 'test_time')]
    logcols = [n for n in voice if any(k in n.lower() for k in ('jitter', 'shimmer', 'nhr'))]
    def arr(split):
        d = data[split]
        return (np.asarray(d['X'], float), np.asarray(d['y'], float),
                np.asarray([str(g) for g in d['groups']]), [str(r) for r in d['row_ids']])
    Xtr, ytr, gtr, rtr = arr('train')
    Xdv, ydv, gdv, rdv = arr('development')
    def patients(X, y, g, tag):
        P = []
        for p in sorted(set(g.tolist())):
            m = g == p
            Xp = X[m].copy()
            for n in logcols:
                j = idx[n]
                Xp[:, j] = np.log(np.maximum(Xp[:, j], 1e-9))
            tt = X[m, idx['test_time']]
            yp = y[m]
            slope = float(np.polyfit(tt, yp, 1)[0]) if (m.sum() > 2 and tt.std() > 0) else 0.0
            P.append(dict(id=p, split=tag, demo=[Xp[:, idx['age']].mean(), Xp[:, idx['sex']].mean()],
                          mean=[Xp[:, idx[n]].mean() for n in voice], std=[Xp[:, idx[n]].std() for n in voice],
                          ymean=float(yp.mean()), ystd=float(yp.std()), slope=slope, n=int(m.sum()),
                          tspan=float(tt.max() - tt.min()), yrows=yp))
        return P
    Ptr = patients(Xtr, ytr, gtr, 'train')
    Pdv = patients(Xdv, ydv, gdv, 'dev')
    Pall = Ptr + Pdv
    def feats(P, kind):
        if kind == 'const':
            return np.zeros((len(P), 0))
        rows = []
        for p in P:
            v = []
            if 'demo' in kind: v += list(p['demo'])
            if 'mean' in kind: v += list(p['mean'])
            if 'std' in kind: v += list(p['std'])
            rows.append(v)
        return np.asarray(rows, float)
    def fit_pred(Ftr, y_, Fte, alpha):
        if Ftr.shape[1] == 0:
            return np.full(len(Fte), float(y_.mean()))
        mu = Ftr.mean(0); sd = Ftr.std(0); sd[sd == 0] = 1.0
        return Ridge(alpha=alpha).fit((Ftr - mu) / sd, y_).predict((Fte - mu) / sd)
    def loo(F, ym, alpha):
        n = len(ym); pred = np.zeros(n)
        for i in range(n):
            m = np.ones(n, bool); m[i] = False
            pred[i] = fit_pred(F[m], ym[m], F[i:i + 1], alpha)[0]
        return pred
    models = [('const', 'const', 0.0), ('demo_a10', 'demo', 10.0), ('mean_a100', 'mean', 100.0),
              ('mean_a1000', 'mean', 1000.0), ('demo_mean_std_a1000', 'demo mean std', 1000.0)]
    def scores(P, pred):
        ym = np.array([p['ymean'] for p in P])
        sq_pat = (ym - pred) ** 2
        sq_rows = np.array([np.sum((p['yrows'] - pred[i]) ** 2) for i, p in enumerate(P)])
        nrows = np.array([p['n'] for p in P], float)
        pat_mae = np.array([np.mean(np.abs(p['yrows'] - pred[i])) for i, p in enumerate(P)])
        return dict(patient_rmse=math.sqrt(sq_pat.mean()), row_rmse=math.sqrt(sq_rows.sum() / nrows.sum()),
                    equal_patient_mae=float(pat_mae.mean())), sq_rows, nrows
    def boot_ci(sq_a, sq_b, nrows, B=1000):
        n = len(nrows); diffs = []
        for _ in range(B):
            s = rng.randint(0, n, n)
            diffs.append(math.sqrt(sq_a[s].sum() / nrows[s].sum()) - math.sqrt(sq_b[s].sum() / nrows[s].sum()))
        return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    out = {'note': 'patient-level aggregate analysis; LOO = leave-one-patient-out on patient means with ridge on standardized aggregates; development labels ARE used here (analysis only), so any downstream selection informed by these numbers is development-tuned, not independent; no evaluation score is produced',
           'n_patients': {'train': len(Ptr), 'dev': len(Pdv)}, 'log_cols': logcols}
    ym_all = np.array([p['ymean'] for p in Pall])
    tot_var = float(np.var(np.concatenate([p['yrows'] for p in Pall])))
    within = float(np.sum([p['n'] * p['ystd'] ** 2 for p in Pall]) / np.sum([p['n'] for p in Pall]))
    sl = [p['slope'] for p in Pall]
    out['target_structure'] = {'between_patient_var_share_all35': (1.0 - within / tot_var) if tot_var > 0 else None,
                               'ymean_sd_across_patients': float(ym_all.std()),
                               'median_within_patient_sd': float(np.median([p['ystd'] for p in Pall])),
                               'slope_per_day_median': float(np.median(sl)),
                               'slope_per_day_iqr': [float(np.percentile(sl, 25)), float(np.percentile(sl, 75))],
                               'median_tspan_days': float(np.median([p['tspan'] for p in Pall]))}
    res = {}
    for setname, P in (('loo_train28', Ptr), ('loo_all35', Pall)):
        ym = np.array([p['ymean'] for p in P]); tab = {}
        base_sq = None; base_n = None
        for name, kind, a in models:
            pred = loo(feats(P, kind), ym, a)
            sc, sq, nr = scores(P, pred)
            if name == 'const':
                base_sq, base_n = sq, nr
            else:
                sc['row_rmse_minus_const_boot95'] = boot_ci(sq, base_sq, nr)
            tab[name] = sc
        res[setname] = tab
    out['patient_held_out'] = res
    ym_tr = np.array([p['ymean'] for p in Ptr]); tr_dev = {}
    for name, kind, a in models:
        pred = fit_pred(feats(Ptr, kind), ym_tr, feats(Pdv, kind), a)
        sc, _, _ = scores(Pdv, pred)
        sc['per_patient_bias'] = [float(pred[i] - p['ymean']) for i, p in enumerate(Pdv)]
        tr_dev[name] = sc
    out['train28_to_dev7_true_groups'] = {'dev_patient_ids': [p['id'] for p in Pdv], 'dev_ymeans': [p['ymean'] for p in Pdv],
                                          'dev_age_sex': [p['demo'] for p in Pdv], 'models': tr_dev}
    perm = {}
    for name, kind, a in (('mean_a1000', 'mean', 1000.0), ('demo_mean_std_a1000', 'demo mean std', 1000.0), ('demo_a10', 'demo', 10.0)):
        F = feats(Pall, kind)
        obs = math.sqrt(np.mean((ym_all - loo(F, ym_all, a)) ** 2))
        cnt = 0; B = 300; null = []
        for _ in range(B):
            yp = rng.permutation(ym_all)
            v = math.sqrt(np.mean((yp - loo(F, yp, a)) ** 2)); null.append(v)
            if v <= obs: cnt += 1
        perm[name] = {'obs_patient_rmse': obs, 'perm_p_leq': (cnt + 1) / (B + 1), 'null_median': float(np.median(null)), 'B': B}
    perm['const_patient_rmse_all35'] = math.sqrt(np.mean((ym_all - loo(feats(Pall, 'const'), ym_all, 0.0)) ** 2))
    out['permutation_all35'] = perm
    Fm = feats(Pall, 'demo mean'); names = ['age', 'sex'] + ['mean_' + n for n in voice]
    rs = [spearmanr(Fm[:, j], ym_all)[0] for j in range(Fm.shape[1])]
    maxnull = []
    for _ in range(1000):
        yp = rng.permutation(ym_all)
        maxnull.append(max(abs(spearmanr(Fm[:, j], yp)[0]) for j in range(Fm.shape[1])))
    order = np.argsort(-np.abs(np.nan_to_num(np.array(rs, float))))
    out['spearman_patient_level_all35'] = {'top': [[names[j], float(rs[j])] for j in order[:8]],
                                           'max_abs_r_null_q95': float(np.percentile(maxnull, 95)),
                                           'max_abs_r_null_q50': float(np.percentile(maxnull, 50))}
    pos = {r: i for i, r in enumerate(rdv)}
    hist = []
    for h in (history or []):
        if not isinstance(h, dict):
            continue
        try:
            pr = h.get('prediction')
            if not h.get('valid') or not pr:
                continue
            pv = np.full(len(rdv), np.nan)
            for r, v in zip(pr.get('row_ids', []), pr.get('prediction', [])):
                if str(r) in pos:
                    pv[pos[str(r)]] = float(v)
            if np.isnan(pv).any():
                continue
            e = pv - ydv
            hist.append({'task': str(h.get('task_id'))[:16], 'action': str(h.get('action_id'))[:12], 'row_rmse': math.sqrt(np.mean(e ** 2)),
                         'per_patient_bias': [float(e[gdv == p['id']].mean()) for p in Pdv],
                         'per_patient_mae': [float(np.abs(e[gdv == p['id']]).mean()) for p in Pdv],
                         'pred_sd_within_patient_median': float(np.median([pv[gdv == p['id']].std() for p in Pdv]))})
        except Exception as ex:
            hist.append({'task': str(h.get('task_id', '?'))[:16], 'error': str(ex)[:80]})
    out['dev_history_breakdown'] = hist[:10]
    out = _clean(out)
    s = json.dumps(out)
    if len(s.encode('utf-8')) > 8000:
        out['dev_history_breakdown'] = [{'task': h.get('task'), 'row_rmse': h.get('row_rmse'), 'per_patient_bias': h.get('per_patient_bias')} for h in out['dev_history_breakdown']]
        s = json.dumps(out)
    if len(s.encode('utf-8')) > 8000:
        out.pop('spearman_patient_level_all35', None); s = json.dumps(out)
    if len(s.encode('utf-8')) > 8000:
        out.pop('dev_history_breakdown', None)
    return out
