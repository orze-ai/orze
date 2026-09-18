import numpy as np
import json
from collections import OrderedDict

ALPHAS = [1.0, 10.0, 100.0, 1000.0]
VARIANTS = ['time', 'voice', 'time+voice']


def _cols(fn):
    ia = fn.index('age')
    isx = fn.index('sex')
    it = fn.index('test_time')
    voice = [i for i in range(len(fn)) if i not in (ia, isx, it)]
    return ia, isx, it, voice


def _build(X, it, voice, log_mask, log_floor, variant):
    V = X[:, voice].copy()
    for j in range(V.shape[1]):
        if log_mask[j]:
            V[:, j] = np.log(np.maximum(V[:, j], log_floor[j]))
    T = X[:, [it]]
    if variant == 'time':
        return T
    if variant == 'voice':
        return V
    return np.hstack([T, V])


def _demean(M, g):
    out = np.empty_like(M)
    for lab in np.unique(g):
        m = (g == lab)
        out[m] = M[m] - M[m].mean(axis=0)
    return out


def _fit(Fd, yd, alpha):
    k = Fd.shape[1]
    return np.linalg.solve(Fd.T @ Fd + alpha * np.eye(k), Fd.T @ yd)


def fit_predict(train, inputs, seed):
    fn = list(train['feature_names'])
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    g = np.asarray([str(v) for v in train['groups']])
    Xe = np.asarray(inputs['X'], dtype=float)
    labs = np.unique(g)
    pmeans = np.array([y[g == l].mean() for l in labs])
    mu = float(pmeans.mean())
    findings = OrderedDict()
    findings['method'] = 'constant (mean of train patient means) + pooled within-patient (fixed-effects) ridge on patient-demeaned test_time and/or log voice features; deployed variant chosen by 28-fold LOPO row RMSE among const and within variants; eval rows demeaned within transductive pseudo-patients keyed by (age, sex) from unlabeled inputs'
    findings['transductive_use'] = 'yes: unlabeled eval X is grouped by (age, sex) and features are demeaned within these pseudo-patients; no eval labels, groups or row IDs are used'
    findings['fallback_used'] = False
    findings['fallbacks'] = []
    findings['n_train_patients'] = int(len(labs))
    findings['constant_mu'] = round(mu, 4)
    pred = np.full(Xe.shape[0], mu)
    try:
        ia, isx, it, voice = _cols(fn)
        vmin = X[:, voice].min(axis=0)
        log_mask = [bool(v > 0) for v in vmin]
        log_floor = [float(vmin[j] * 0.5) if log_mask[j] else 1.0 for j in range(len(voice))]
        findings['log_voice_columns'] = [fn[voice[j]] for j in range(len(voice)) if log_mask[j]]
        feats = {v: _build(X, it, voice, log_mask, log_floor, v) for v in VARIANTS}
        names = {'time': ['test_time'], 'voice': [fn[i] for i in voice], 'time+voice': ['test_time'] + [fn[i] for i in voice]}
        yd_full = _demean(y[:, None], g)[:, 0]
        findings['train_within_patient_y_sd'] = round(float(yd_full.std()), 4)
        findings['train_between_patient_mean_sd'] = round(float(pmeans.std()), 4)

        def lopo(variant, alpha):
            se = 0.0
            n = 0
            maes = []
            for l in labs:
                te = (g == l)
                tr = ~te
                mu_o = float(pmeans[labs != l].mean())
                if variant == 'const':
                    p = np.full(int(te.sum()), mu_o)
                else:
                    F = feats[variant]
                    Fd_tr = _demean(F[tr], g[tr])
                    s = Fd_tr.std(axis=0)
                    s[s == 0] = 1.0
                    yd_tr = _demean(y[tr][:, None], g[tr])[:, 0]
                    beta = _fit(Fd_tr / s, yd_tr, alpha)
                    Fd_te = (F[te] - F[te].mean(axis=0)) / s
                    p = np.clip(mu_o + Fd_te @ beta, y[tr].min(), y[tr].max())
                r = y[te] - p
                se += float((r ** 2).sum())
                n += int(te.sum())
                maes.append(float(np.abs(r).mean()))
            return (se / n) ** 0.5, float(np.mean(maes))

        table = OrderedDict()
        rm, ma = lopo('const', None)
        table['const'] = {'row_rmse': round(rm, 4), 'equal_patient_mae': round(ma, 4)}
        best = ('const', None, rm)
        for v in VARIANTS:
            for a in ALPHAS:
                rm, ma = lopo(v, a)
                table[v + '_alpha' + format(a, 'g')] = {'row_rmse': round(rm, 4), 'equal_patient_mae': round(ma, 4)}
                if rm < best[2]:
                    best = (v, a, rm)
        findings['train_lopo_28'] = table
        findings['selected_variant'] = best[0]
        findings['selected_alpha'] = best[1]
        findings['lopo_row_rmse_delta_selected_minus_const'] = round(best[2] - table['const']['row_rmse'], 4)
        if best[0] == 'const':
            pred = np.full(Xe.shape[0], mu)
            findings['deployed'] = 'constant only; LOPO did not prefer any within-patient variant'
        else:
            F = feats[best[0]]
            Fd = _demean(F, g)
            s = Fd.std(axis=0)
            s[s == 0] = 1.0
            beta = _fit(Fd / s, yd_full, best[1])
            findings['beta_per_within_sd'] = {names[best[0]][j]: round(float(beta[j]), 4) for j in range(len(beta))}
            res = yd_full - (Fd / s) @ beta
            findings['train_within_r2'] = round(float(1.0 - res.var() / yd_full.var()), 4)
            keys = [(round(float(Xe[i, ia]), 3), round(float(Xe[i, isx]), 3)) for i in range(Xe.shape[0])]
            kmap = {}
            ge = np.array([kmap.setdefault(k, len(kmap)) for k in keys])
            sizes = [int((ge == k).sum()) for k in range(len(kmap))]
            findings['eval_pseudo_groups'] = {'n': len(kmap), 'sizes': sizes, 'n_singletons': int(sum(1 for z in sizes if z == 1)), 'n_unique_age': int(len(np.unique(Xe[:, ia])))}
            Fe = _build(Xe, it, voice, log_mask, log_floor, best[0])
            Fde = _demean(Fe, ge) / s
            pred = np.clip(mu + Fde @ beta, y.min(), y.max())
            findings['deployed'] = 'constant + within-pseudo-patient ' + best[0] + ' ridge alpha ' + format(best[1], 'g')
        findings['eval_pred_range'] = [round(float(pred.min()), 4), round(float(pred.max()), 4)]
        findings['eval_pred_sd'] = round(float(pred.std()), 4)
    except Exception as e:
        pred = np.full(Xe.shape[0], mu)
        findings['fallback_used'] = True
        findings['fallbacks'].append('constant_mu_fallback: ' + type(e).__name__ + ': ' + str(e)[:200])
    pred = np.where(np.isfinite(pred), pred, mu)
    return {'prediction': [float(v) for v in pred], 'findings': json.loads(json.dumps(findings))}
