import json
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge

CAND_KEYS = ['row_median', 'row_mean', 'pat_mean_mean', 'pat_mean_median', 'pat_med_median', 'pat_mean_trim10']


def _trimmed(v, frac=0.1):
    v = np.sort(np.asarray(v, dtype=float))
    k = int(np.floor(v.size * frac))
    if 2 * k >= v.size:
        return float(np.median(v))
    return float(np.mean(v[k:v.size - k]))


def _cands(y, g):
    d = defaultdict(list)
    for yi, gi in zip(y, g):
        d[gi].append(float(yi))
    pm = np.array([np.mean(v) for v in d.values()], dtype=float)
    pmed = np.array([np.median(v) for v in d.values()], dtype=float)
    return {
        'row_median': float(np.median(y)),
        'row_mean': float(np.mean(y)),
        'pat_mean_mean': float(np.mean(pm)),
        'pat_mean_median': float(np.median(pm)),
        'pat_med_median': float(np.median(pmed)),
        'pat_mean_trim10': _trimmed(pm, 0.1),
    }


def _std_fit(Z):
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def fit_predict(train, inputs, seed):
    findings = {
        'fallback_used': False,
        'design': 'LOPO-selected patient-equal-weighted constant plus permutation-gated patient-level shrunk ridge on (age,sex) pseudo-patient mean features',
        'transductive_note': 'unlabeled evaluation X used only to group rows by exact (age,sex) and to average features within those pseudo-patient groups; no evaluation labels used'
    }
    Xe = np.asarray(inputs['X'], dtype=float)
    n_eval = int(Xe.shape[0])
    y_all = np.asarray(train['y'], dtype=float)
    safe_const = float(np.median(y_all))
    try:
        X = np.asarray(train['X'], dtype=float)
        y = y_all
        g = np.array([str(v) for v in train['groups']])
        names = [str(v) for v in train['feature_names']]
        ename = [str(v) for v in inputs['feature_names']]
        pats = sorted(set(g.tolist()))
        npat = len(pats)
        n = float(y.size)
        findings['n_train_patients'] = npat

        sse = {k: 0.0 for k in CAND_KEYS}
        pmae = {k: [] for k in CAND_KEYS}
        for p in pats:
            m = (g != p)
            c = _cands(y[m], g[m])
            yo = y[~m]
            for k in CAND_KEYS:
                r = yo - c[k]
                sse[k] += float(np.sum(r * r))
                pmae[k].append(float(np.mean(np.abs(r))))
        findings['lopo_constant_rmse_gmae_by_rule'] = {
            k: [round((sse[k] / n) ** 0.5, 4), round(float(np.mean(pmae[k])), 4)] for k in CAND_KEYS}
        best_key = min(CAND_KEYS, key=lambda k: sse[k])
        full_c = _cands(y, g)
        const_value = float(full_c[best_key])
        const_rmse = (sse[best_key] / n) ** 0.5
        findings['constant_rule_selected'] = best_key
        findings['constant_value'] = round(const_value, 4)
        findings['lopo_constant_only_rmse'] = round(const_rmse, 4)

        fidx = [i for i, nm in enumerate(names) if nm != 'test_time']
        try:
            fidx_e = [ename.index(names[i]) for i in fidx]
        except Exception:
            fidx_e = list(fidx)
            findings['feature_name_alignment'] = 'fell_back_to_positional'
        P = np.array([np.mean(X[g == p][:, fidx], axis=0) for p in pats], dtype=float)
        ybar = np.array([float(np.mean(y[g == p])) for p in pats], dtype=float)
        cnt = np.array([float(np.sum(g == p)) for p in pats], dtype=float)
        wss = float(np.sum([np.sum((y[g == p] - ybar[j]) ** 2) for j, p in enumerate(pats)]))
        alphas = [30.0, 100.0, 300.0, 1000.0, 3000.0]
        ws = [0.25, 0.5, 0.75, 1.0]

        cfold = np.zeros(npat, dtype=float)
        for j, p in enumerate(pats):
            m = (g != p)
            cfold[j] = float(_cands(y[m], g[m])[best_key])

        def lopo_pred(t, a):
            pr = np.zeros(npat, dtype=float)
            bs = np.zeros(npat, dtype=float)
            for j in range(npat):
                m = np.ones(npat, dtype=bool)
                m[j] = False
                mu, sd = _std_fit(P[m])
                rg = Ridge(alpha=a, fit_intercept=True)
                rg.fit((P[m] - mu) / sd, t[m])
                pr[j] = float(rg.predict(((P[j] - mu) / sd).reshape(1, -1))[0])
                bs[j] = float(np.mean(t[m]))
            return pr, bs

        table = []
        best = None
        cache = {}
        for a in alphas:
            pr, bs = lopo_pred(ybar, a)
            cache[a] = (pr, bs)
            off = pr - bs
            for w in ws:
                pv = cfold + w * off
                se = float(np.sum(cnt * ((pv - ybar) ** 2))) + wss
                rmse = (se / n) ** 0.5
                gm_approx = float(np.mean(np.abs(pv - ybar)))
                table.append([round(rmse, 4), round(gm_approx, 4), a, w])
                if best is None or rmse < best[0]:
                    best = (rmse, a, w)
        table.sort(key=lambda t: t[0])
        findings['lopo_feature_top6_rmse_gmaeapprox_alpha_w'] = table[:6]
        rmse_feat, a_best, w_best = best[0], best[1], best[2]
        findings['lopo_feature_best'] = [round(rmse_feat, 4), a_best, w_best]

        def lopo_delta(t, a):
            pr, bs = lopo_pred(t, a)
            sc = float(np.sum(cnt * ((bs - t) ** 2)))
            sr = float(np.sum(cnt * ((pr - t) ** 2)))
            return sc - sr

        obs_delta = lopo_delta(ybar, a_best)
        rng = np.random.default_rng(int(seed) if seed is not None else 0)
        nperm = 200
        ge = 0
        for _ in range(nperm):
            yp = ybar[rng.permutation(npat)]
            if lopo_delta(yp, a_best) >= obs_delta:
                ge += 1
        p_perm = (ge + 1.0) / (nperm + 1.0)
        findings['permutation'] = {'n_perm': nperm, 'alpha': a_best,
                                  'observed_sse_gain_over_fold_mean': round(obs_delta, 3),
                                  'p_value': round(p_perm, 4)}
        use_feat = bool((rmse_feat < const_rmse - 1e-9) and (p_perm <= 0.10) and (obs_delta > 0.0))
        findings['feature_term_active'] = use_feat

        ia = ename.index('age') if 'age' in ename else 0
        isx = ename.index('sex') if 'sex' in ename else 1
        keys = {}
        for i in range(n_eval):
            k = (round(float(Xe[i, ia]), 6), round(float(Xe[i, isx]), 6))
            keys.setdefault(k, []).append(i)
        findings['n_pseudo_groups_eval'] = len(keys)
        findings['pseudo_group_sizes_desc'] = sorted([len(v) for v in keys.values()], reverse=True)[:20]

        pred = np.full(n_eval, const_value, dtype=float)
        if use_feat:
            mu, sd = _std_fit(P)
            rg = Ridge(alpha=a_best, fit_intercept=True)
            rg.fit((P - mu) / sd, ybar)
            m0 = float(np.mean(ybar))
            gpred = []
            for k, idxs in keys.items():
                ii = np.array(idxs, dtype=int)
                fm = np.mean(Xe[ii][:, fidx_e], axis=0)
                pv = float(rg.predict(((fm - mu) / sd).reshape(1, -1))[0])
                val = const_value + w_best * (pv - m0)
                pred[ii] = val
                gpred.append(round(val, 3))
            findings['pseudo_group_predictions_sorted'] = sorted(gpred)
        lo = float(np.min(y)) - 5.0
        hi = float(np.max(y)) + 5.0
        pred = np.clip(pred, lo, hi)
        pred = np.where(np.isfinite(pred), pred, const_value)
        findings['pred_summary'] = {'n': int(pred.size), 'mean': round(float(np.mean(pred)), 3),
                                   'sd': round(float(np.std(pred)), 4),
                                   'min': round(float(np.min(pred)), 3),
                                   'max': round(float(np.max(pred)), 3)}
        out = [float(v) for v in pred]
    except Exception as e:
        findings = {'fallback_used': True,
                    'fallback_reason': str(e)[:300],
                    'fallback_rule': 'train row median constant',
                    'design': findings.get('design', ''),
                    'transductive_note': 'none used in fallback path'}
        out = [safe_const] * n_eval
    txt = json.dumps(findings)
    if len(txt.encode('utf-8')) > 8000:
        findings = {'fallback_used': findings.get('fallback_used', False),
                    'truncated': True,
                    'constant_rule_selected': findings.get('constant_rule_selected'),
                    'feature_term_active': findings.get('feature_term_active'),
                    'permutation': findings.get('permutation'),
                    'pred_summary': findings.get('pred_summary')}
    return {'prediction': out, 'findings': findings}
