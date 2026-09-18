import numpy as np
import json
from sklearn.ensemble import HistGradientBoostingRegressor

BRANCHES = ['squared', 'deflate_drop', 'deflate_keep']
HGB = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1)
B_CLIP = 0.004


def _growth(days, y):
    ud, inv = np.unique(days, return_inverse=True)
    tot = np.bincount(inv, weights=y, minlength=len(ud))
    keep = tot > 0
    ud = ud[keep]
    tot = tot[keep]
    if len(ud) < 30:
        return 0.0, int(len(ud)), 'too_few_dates'
    w = 2.0 * np.pi * ud / 365.25
    A = np.column_stack([np.ones_like(ud), ud, np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
    coef = np.linalg.lstsq(A, np.log(tot), rcond=None)[0]
    b_raw = float(coef[1])
    b = float(np.clip(b_raw, -B_CLIP, B_CLIP))
    return b, int(len(ud)), ('clipped' if b != b_raw else 'ok')


def _fit_pred(branch, Xtr, ytr, Xte, di, yi, seed, info):
    if branch == 'squared':
        m = HistGradientBoostingRegressor(loss='squared_error', random_state=seed, **HGB)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    b, nd, status = _growth(Xtr[:, di], ytr)
    ref = float(np.max(Xtr[:, di]))
    gtr = np.exp(b * (Xtr[:, di] - ref))
    gte = np.exp(b * (Xte[:, di] - ref))
    info[branch] = {'growth_b_per_day': b, 'annual_factor': float(np.exp(365.0 * b)), 'n_dates': nd,
                    'status': status, 'eval_factor_range': [float(np.min(gte)), float(np.max(gte))]}
    if branch == 'deflate_drop':
        keep = [j for j in range(Xtr.shape[1]) if j not in (di, yi)]
        Xtr2, Xte2 = Xtr[:, keep], Xte[:, keep]
    elif branch == 'deflate_keep':
        Xtr2, Xte2 = Xtr, Xte
    else:
        raise ValueError('unknown branch ' + str(branch))
    m = HistGradientBoostingRegressor(loss='squared_error', random_state=seed, **HGB)
    m.fit(Xtr2, ytr / gtr)
    return m.predict(Xte2) * gte


def _rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _date_mae(p, y, days):
    vals = []
    for d in np.unique(days):
        idx = days == d
        vals.append(float(np.mean(np.abs(p[idx] - y[idx]))))
    return float(np.mean(vals)) if vals else float('nan')


def fit_predict(train, inputs, seed):
    fn = list(train['feature_names'])
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xe = np.asarray(inputs['X'], dtype=float)
    di = fn.index('elapsed_day')
    yi = fn.index('yr')
    days = X[:, di]
    dmax = float(np.max(days))
    findings = {'branches': BRANCHES, 'folds': [], 'fold_rmse': {b: [] for b in BRANCHES},
                'fold_date_mae': {b: [] for b in BRANCHES}, 'fold_growth': [], 'errors': {}, 'fallback': None,
                'note': 'growth rate b fitted on log daily totals of training dates only (trend + 2 harmonics); target deflated by exp(b*(day-last_train_day)) and predictions re-inflated; branch selection by mean RMSE over two forward 61-day training holdouts; HGB hyperparameters fixed at baseline; evaluation covariates used only for final prediction'}
    for k in range(2):
        hi = dmax - 61.0 * k
        lo = hi - 61.0
        te = (days > lo) & (days <= hi)
        tr = days <= lo
        if te.sum() < 100 or tr.sum() < 500:
            findings['folds'].append({'fold': k + 1, 'skipped': True})
            continue
        findings['folds'].append({'fold': k + 1, 'holdout_day_range': [float(lo), float(hi)],
                                  'holdout_rows': int(te.sum()), 'train_rows': int(tr.sum())})
        info = {}
        for b in BRANCHES:
            try:
                p = _fit_pred(b, X[tr], y[tr], X[te], di, yi, seed, info)
                p = np.where(np.isfinite(p), p, np.median(y[tr]))
                p = np.maximum(p, 0.0)
                findings['fold_rmse'][b].append(_rmse(p, y[te]))
                findings['fold_date_mae'][b].append(_date_mae(p, y[te], days[te]))
            except Exception as exc:
                findings['errors'][b + '_fold' + str(k + 1)] = type(exc).__name__
                findings['fold_rmse'][b].append(float('inf'))
                findings['fold_date_mae'][b].append(float('inf'))
        findings['fold_growth'].append(info)
    mean_rmse = {}
    mean_mae = {}
    for b in BRANCHES:
        r = findings['fold_rmse'][b]
        m = findings['fold_date_mae'][b]
        mean_rmse[b] = float(np.mean(r)) if r else float('inf')
        mean_mae[b] = float(np.mean(m)) if m else float('inf')
    order = sorted(BRANCHES, key=lambda b: (mean_rmse[b], BRANCHES.index(b)))
    selected = order[0] if np.isfinite(mean_rmse[order[0]]) else 'squared'
    findings['mean_forward_rmse'] = {b: (mean_rmse[b] if np.isfinite(mean_rmse[b]) else None) for b in BRANCHES}
    findings['mean_forward_date_mae'] = {b: (mean_mae[b] if np.isfinite(mean_mae[b]) else None) for b in BRANCHES}
    findings['selected_branch'] = selected
    findings['best_by_date_mae'] = min(BRANCHES, key=lambda b: (mean_mae[b], BRANCHES.index(b)))
    findings['train_rows'] = int(X.shape[0])
    findings['train_day_range'] = [float(np.min(days)), dmax]
    findings['eval_day_range'] = [float(np.min(Xe[:, di])), float(np.max(Xe[:, di]))]
    final_info = {}
    try:
        pred = _fit_pred(selected, X, y, Xe, di, yi, seed, final_info)
    except Exception as exc:
        findings['errors']['final_' + selected] = type(exc).__name__
        findings['fallback'] = 'squared_error_after_final_fit_failure'
        final_info = {}
        pred = _fit_pred('squared', X, y, Xe, di, yi, seed, final_info)
    findings['final_growth'] = final_info
    pred = np.asarray(pred, dtype=float)
    bad = ~np.isfinite(pred)
    findings['nonfinite_replaced'] = int(bad.sum())
    if bad.any():
        pred[bad] = float(np.median(y))
    pred = np.maximum(pred, 0.0)
    findings['pred_summary'] = {'mean': float(np.mean(pred)), 'max': float(np.max(pred)), 'min': float(np.min(pred))}
    return {'prediction': pred.tolist(), 'findings': json.loads(json.dumps(findings))}
