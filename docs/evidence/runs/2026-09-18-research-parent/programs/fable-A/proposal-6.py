import json
import math
import numpy as np
from scipy import stats


def _patient_table(X, y, groups, fn):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(groups)
    ia = fn.index('age')
    isx = fn.index('sex')
    ids = list(dict.fromkeys(g.tolist()))
    A = np.array([X[g == p, ia].mean() for p in ids])
    S = np.array([X[g == p, isx].mean() for p in ids])
    M = np.array([y[g == p].mean() for p in ids])
    Y = [y[g == p] for p in ids]
    return ids, A, S, M, Y


def _ridge(F, t, alpha):
    mu = float(t.mean())
    p = F.shape[1]
    w = np.linalg.solve(F.T @ F + alpha * np.eye(p), F.T @ (t - mu))
    return mu, w


def _lopo(F, M, Y, alpha):
    n = len(M)
    sse = 0.0
    nrow = 0
    pmae = []
    preds = []
    for k in range(n):
        tr = np.ones(n, dtype=bool)
        tr[k] = False
        if F.shape[1] > 0:
            mean_ = F[tr].mean(0)
            sd_ = F[tr].std(0)
            sd_ = np.where(sd_ > 0, sd_, 1.0)
            Fz = (F - mean_) / sd_
            mu, w = _ridge(Fz[tr], M[tr], alpha)
            pred = mu + float(Fz[k] @ w)
        else:
            pred = float(M[tr].mean())
        preds.append(pred)
        r = pred - Y[k]
        sse += float((r ** 2).sum())
        nrow += len(Y[k])
        pmae.append(float(np.abs(r).mean()))
    return {'row_rmse': round(math.sqrt(sse / nrow), 4), 'equal_patient_mae': round(float(np.mean(pmae)), 4)}, np.array(preds)


def _perm_p(a, b, rng, nperm=2000):
    obs = stats.spearmanr(a, b).correlation
    if not np.isfinite(obs):
        return None, None
    cnt = 0
    for _ in range(nperm):
        r = stats.spearmanr(a, rng.permutation(b)).correlation
        if np.isfinite(r) and abs(r) >= abs(obs):
            cnt += 1
    return round(float(obs), 4), round((cnt + 1) / (nperm + 1), 4)


def fit_predict(train, inputs, seed=0):
    findings = {'fallback_used': False, 'fallbacks': [],
                'method': 'constant (mean of train patient means) + patient-level ridge on standardized age fitted on 28 patient means; alpha by 28-fold LOPO row RMSE; no voice features, no test_time term, no transductive use of inputs; sex variants are LOPO diagnostics only, not deployed'}
    rng = np.random.RandomState(int(seed) if seed is not None else 0)
    fn = list(train['feature_names'])
    ids, A, S, M, Y = _patient_table(train['X'], train['y'], train['groups'], fn)
    ytr = np.asarray(train['y'], dtype=float)
    const = float(M.mean())
    alphas = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]
    variants = {'age': np.column_stack([A]), 'sex': np.column_stack([S]), 'age+sex': np.column_stack([A, S])}
    lopo = {}
    lopo['const'], _ = _lopo(np.zeros((len(M), 0)), M, Y, 0.0)
    best = None
    for name, F in variants.items():
        for al in alphas:
            res, _ = _lopo(F, M, Y, al)
            lopo[name + '_alpha' + ('%g' % al)] = res
            if name == 'age' and (best is None or res['row_rmse'] < best[1]['row_rmse']):
                best = (al, res)
    alpha_sel = float(best[0])
    F = variants['age']
    mean_ = F.mean(0)
    sd_ = F.std(0)
    sd_ = np.where(sd_ > 0, sd_, 1.0)
    mu, w = _ridge((F - mean_) / sd_, M, alpha_sel)
    Xe = np.asarray(inputs['X'], dtype=float)
    ia = fn.index('age')
    Fe = (Xe[:, [ia]] - mean_) / sd_
    pred = mu + Fe @ w
    lo, hi = float(ytr.min()), float(ytr.max())
    nclip = int(((pred < lo) | (pred > hi)).sum())
    pred = np.clip(pred, lo, hi)
    if not np.all(np.isfinite(pred)):
        findings['fallback_used'] = True
        findings['fallbacks'].append('non-finite predictions replaced by train constant')
        pred = np.where(np.isfinite(pred), pred, const)
    sp_age, p_age = _perm_p(A, M, rng)
    sp_sex, p_sex = _perm_p(S, M, rng)
    findings.update({
        'n_train_patients': int(len(ids)),
        'constant_mean_of_patient_means': round(const, 4),
        'intercept_mu': round(mu, 4),
        'alpha_selected_by_lopo': alpha_sel,
        'age_coef_per_sd': round(float(w[0]), 4),
        'age_coef_per_year': round(float(w[0] / sd_[0]), 4),
        'train_age_mean_sd': [round(float(mean_[0]), 3), round(float(sd_[0]), 3)],
        'train_age_range': [float(A.min()), float(A.max())],
        'eval_age_range': [float(Xe[:, ia].min()), float(Xe[:, ia].max())],
        'eval_unique_age_values': int(len(np.unique(Xe[:, ia]))),
        'patient_level_spearman_age_vs_mean_motor': sp_age,
        'patient_level_perm_p_age': p_age,
        'patient_level_spearman_sex_vs_mean_motor': sp_sex,
        'patient_level_perm_p_sex': p_sex,
        'patient_level_pearson_age': round(float(np.corrcoef(A, M)[0, 1]), 4),
        'train_lopo_28': lopo,
        'lopo_row_rmse_delta_age_minus_const': round(best[1]['row_rmse'] - lopo['const']['row_rmse'], 4),
        'eval_pred_range': [round(float(pred.min()), 3), round(float(pred.max()), 3)],
        'n_clipped_to_train_range': nclip,
        'note': 'one-factor change vs the train-patient-mean constant: only a patient-level age term is added; dev labels are not used; sex/age+sex LOPO rows are diagnostics only'
    })
    return {'prediction': [float(v) for v in pred], 'findings': findings}
