import numpy as np
import json
from sklearn.ensemble import HistGradientBoostingRegressor

BRANCHES = ['squared', 'poisson', 'log1p']


def _make(loss, seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1, loss=loss, random_state=seed)


def _fit_pred(branch, Xtr, ytr, Xte, seed):
    if branch == 'squared':
        m = _make('squared_error', seed)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    if branch == 'poisson':
        m = _make('poisson', seed)
        m.fit(Xtr, np.maximum(ytr, 0.0))
        return m.predict(Xte)
    if branch == 'log1p':
        m = _make('squared_error', seed)
        m.fit(Xtr, np.log1p(np.maximum(ytr, 0.0)))
        return np.expm1(m.predict(Xte))
    raise ValueError('unknown branch ' + str(branch))


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
    days = X[:, di]
    dmax = float(np.max(days))
    findings = {'branches': BRANCHES, 'folds': [], 'fold_rmse': {b: [] for b in BRANCHES},
                'fold_date_mae': {b: [] for b in BRANCHES}, 'errors': {}, 'fallback': None,
                'note': 'branch selection uses only training dates (two forward 61-day holdouts); evaluation covariates used only for final prediction; baseline HGB hyperparameters fixed'}
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
        for b in BRANCHES:
            try:
                p = _fit_pred(b, X[tr], y[tr], X[te], seed)
                p = np.where(np.isfinite(p), p, np.median(y[tr]))
                p = np.maximum(p, 0.0)
                findings['fold_rmse'][b].append(_rmse(p, y[te]))
                findings['fold_date_mae'][b].append(_date_mae(p, y[te], days[te]))
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
    selected = order[0] if np.isfinite(mean_rmse[order[0]]) else 'squared'
    findings['mean_forward_rmse'] = {b: (mean_rmse[b] if np.isfinite(mean_rmse[b]) else None) for b in BRANCHES}
    findings['mean_forward_date_mae'] = {b: (mean_mae[b] if np.isfinite(mean_mae[b]) else None) for b in BRANCHES}
    findings['selected_branch'] = selected
    findings['best_by_date_mae'] = min(BRANCHES, key=lambda b: (mean_mae[b], BRANCHES.index(b)))
    findings['train_rows'] = int(X.shape[0])
    findings['train_day_range'] = [float(np.min(days)), dmax]
    findings['eval_day_range'] = [float(np.min(Xe[:, di])), float(np.max(Xe[:, di]))]
    try:
        pred = _fit_pred(selected, X, y, Xe, seed)
    except Exception as exc:
        findings['errors']['final_' + selected] = type(exc).__name__
        findings['fallback'] = 'squared_error_after_final_fit_failure'
        pred = _fit_pred('squared', X, y, Xe, seed)
    pred = np.asarray(pred, dtype=float)
    bad = ~np.isfinite(pred)
    findings['nonfinite_replaced'] = int(bad.sum())
    if bad.any():
        pred[bad] = float(np.median(y))
    pred = np.maximum(pred, 0.0)
    findings['pred_summary'] = {'mean': float(np.mean(pred)), 'max': float(np.max(pred)), 'min': float(np.min(pred))}
    return {'prediction': pred.tolist(), 'findings': json.loads(json.dumps(findings))}
