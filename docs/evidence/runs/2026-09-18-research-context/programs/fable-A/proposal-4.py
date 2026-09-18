import numpy as np
import json
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor

BRANCHES = ['base', 'lag_point', 'lag_smooth']
LAG = 364


def _make(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1, loss='squared_error', random_state=seed)


def _tables(days, hrs, y):
    hour_tab = {}
    day_tot = defaultdict(float)
    for d, h, v in zip(days, hrs, y):
        hour_tab[(int(round(d)), int(round(h)))] = float(v)
        day_tot[int(round(d))] += float(v)
    return hour_tab, dict(day_tot)


def _lag_feats(days, hrs, hour_tab, day_tot, smooth):
    n = len(days)
    f_hr = np.full(n, np.nan)
    f_day = np.full(n, np.nan)
    for i in range(n):
        d = int(round(days[i])) - LAG
        h = int(round(hrs[i]))
        if smooth:
            hv = [hour_tab[(dd, h)] for dd in (d - 7, d, d + 7) if (dd, h) in hour_tab]
            dv = [day_tot[dd] for dd in range(d - 3, d + 4) if dd in day_tot]
            if len(hv) >= 2:
                f_hr[i] = float(np.mean(hv))
            if len(dv) >= 4:
                f_day[i] = float(np.mean(dv))
        else:
            if (d, h) in hour_tab:
                f_hr[i] = hour_tab[(d, h)]
            if d in day_tot:
                f_day[i] = day_tot[d]
    return np.column_stack([f_hr, f_day])


def _features(branch, Xtr, ytr, Xte, di, hi):
    if branch == 'base':
        return Xtr, Xte, None
    smooth = branch == 'lag_smooth'
    hour_tab, day_tot = _tables(Xtr[:, di], Xtr[:, hi], ytr)
    ftr = _lag_feats(Xtr[:, di], Xtr[:, hi], hour_tab, day_tot, smooth)
    fte = _lag_feats(Xte[:, di], Xte[:, hi], hour_tab, day_tot, smooth)
    info = {'train_lag_hr_available_frac': float(np.mean(np.isfinite(ftr[:, 0]))),
            'eval_lag_hr_available_frac': float(np.mean(np.isfinite(fte[:, 0]))),
            'eval_lag_day_available_frac': float(np.mean(np.isfinite(fte[:, 1])))}
    return np.hstack([Xtr, ftr]), np.hstack([Xte, fte]), info


def _fit_pred(branch, Xtr, ytr, Xte, seed, di, hi):
    A, B, info = _features(branch, Xtr, ytr, Xte, di, hi)
    m = _make(seed)
    m.fit(A, ytr)
    return m.predict(B), info


def _rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _date_mae(p, y, days):
    vals = []
    for d in np.unique(days):
        idx = days == d
        vals.append(float(np.mean(np.abs(p[idx] - y[idx]))))
    return float(np.mean(vals)) if vals else float('nan')


def _yoy_diag(days, mnth, yrs, y):
    hour_tab, day_tot = _tables(days, np.zeros_like(days), y)
    out = {}
    cur = defaultdict(float)
    prev = defaultdict(float)
    seen = set()
    for d, mo, yv in zip(days, mnth, yrs):
        dd = int(round(d))
        if yv < 0.5 or dd in seen:
            continue
        seen.add(dd)
        if dd in day_tot and (dd - LAG) in day_tot:
            cur[int(round(mo))] += day_tot[dd]
            prev[int(round(mo))] += day_tot[dd - LAG]
    for mo in sorted(cur):
        if prev[mo] > 0:
            out['month_' + str(mo)] = float(cur[mo] / prev[mo])
    tc = sum(cur.values())
    tp = sum(prev.values())
    out['overall'] = float(tc / tp) if tp > 0 else None
    return out


def fit_predict(train, inputs, seed):
    fn = list(train['feature_names'])
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xe = np.asarray(inputs['X'], dtype=float)
    di = fn.index('elapsed_day')
    hi = fn.index('hr')
    mi = fn.index('mnth')
    yi = fn.index('yr')
    days = X[:, di]
    dmax = float(np.max(days))
    findings = {'branches': BRANCHES, 'lag_days': LAG, 'folds': [],
                'fold_rmse': {b: [] for b in BRANCHES}, 'fold_date_mae': {b: [] for b in BRANCHES},
                'fold_lag_info': {b: [] for b in BRANCHES if b != 'base'}, 'errors': {}, 'fallback': None,
                'note': 'lag features are training-label counts at elapsed_day-364 same hr (point) or mean of lag 357/364/371 same hr plus mean daily total over lag week (smooth); NaN when unavailable; tables built only from the fold/final training labels; branch selection by mean RMSE over two forward 61-day training holdouts; fold 2 training rows have no lag so its lag branches cannot learn lag relations; HGB hyperparameters fixed at baseline; evaluation covariates used only for final prediction'}
    try:
        findings['train_yoy_daily_total_ratio_2012_vs_2011'] = _yoy_diag(days, X[:, mi], X[:, yi], y)
    except Exception as exc:
        findings['errors']['yoy_diag'] = type(exc).__name__
    for k in range(2):
        hi_d = dmax - 61.0 * k
        lo_d = hi_d - 61.0
        te = (days > lo_d) & (days <= hi_d)
        tr = days <= lo_d
        if te.sum() < 100 or tr.sum() < 500:
            findings['folds'].append({'fold': k + 1, 'skipped': True})
            continue
        findings['folds'].append({'fold': k + 1, 'holdout_day_range': [float(lo_d), float(hi_d)],
                                  'holdout_rows': int(te.sum()), 'train_rows': int(tr.sum())})
        for b in BRANCHES:
            try:
                p, info = _fit_pred(b, X[tr], y[tr], X[te], seed, di, hi)
                p = np.where(np.isfinite(p), p, np.median(y[tr]))
                p = np.maximum(p, 0.0)
                findings['fold_rmse'][b].append(_rmse(p, y[te]))
                findings['fold_date_mae'][b].append(_date_mae(p, y[te], days[te]))
                if info is not None:
                    findings['fold_lag_info'][b].append(info)
            except Exception as exc:
                findings['errors'][b + '_fold' + str(k + 1)] = type(exc).__name__
                findings['fold_rmse'][b].append(float('inf'))
                findings['fold_date_mae'][b].append(float('inf'))
    mean_rmse = {}
    mean_mae = {}
    for b in BRANCHES:
        r = findings['fold_rmse'][b]
        m = findings['fold_date_mae'][b]
        mean_rmse[b] = float(np.mean(r)) if r else float('inf')
        mean_mae[b] = float(np.mean(m)) if m else float('inf')
    order = sorted(BRANCHES, key=lambda b: (mean_rmse[b], BRANCHES.index(b)))
    selected = order[0] if np.isfinite(mean_rmse[order[0]]) else 'base'
    findings['mean_forward_rmse'] = {b: (mean_rmse[b] if np.isfinite(mean_rmse[b]) else None) for b in BRANCHES}
    findings['mean_forward_date_mae'] = {b: (mean_mae[b] if np.isfinite(mean_mae[b]) else None) for b in BRANCHES}
    findings['fold1_rmse_by_branch'] = {b: (findings['fold_rmse'][b][0] if findings['fold_rmse'][b] and np.isfinite(findings['fold_rmse'][b][0]) else None) for b in BRANCHES}
    findings['selected_branch'] = selected
    findings['best_by_date_mae'] = min(BRANCHES, key=lambda b: (mean_mae[b], BRANCHES.index(b)))
    findings['train_rows'] = int(X.shape[0])
    findings['train_day_range'] = [float(np.min(days)), dmax]
    findings['eval_day_range'] = [float(np.min(Xe[:, di])), float(np.max(Xe[:, di]))]
    try:
        pred, info = _fit_pred(selected, X, y, Xe, seed, di, hi)
        findings['final_lag_info'] = info
    except Exception as exc:
        findings['errors']['final_' + selected] = type(exc).__name__
        findings['fallback'] = 'base_after_final_fit_failure'
        pred, _ = _fit_pred('base', X, y, Xe, seed, di, hi)
    pred = np.asarray(pred, dtype=float)
    bad = ~np.isfinite(pred)
    findings['nonfinite_replaced'] = int(bad.sum())
    if bad.any():
        pred[bad] = float(np.median(y))
    pred = np.maximum(pred, 0.0)
    findings['pred_summary'] = {'mean': float(np.mean(pred)), 'max': float(np.max(pred)), 'min': float(np.min(pred))}
    return {'prediction': pred.tolist(), 'findings': json.loads(json.dumps(findings))}
