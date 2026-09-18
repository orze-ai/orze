import numpy as np
import json
import collections
from sklearn.ensemble import HistGradientBoostingRegressor

CONFIGS = {
    'base': {'max_iter': 250, 'learning_rate': 0.08, 'max_leaf_nodes': 31, 'l2_regularization': 1.0, 'min_samples_leaf': 20},
    'slow': {'max_iter': 600, 'learning_rate': 0.04, 'max_leaf_nodes': 31, 'l2_regularization': 1.0, 'min_samples_leaf': 40},
}
BRANCHES = ['base', 'slow', 'base_bag', 'slow_bag']
N_BAGS = 6


def _make(cfg, seed):
    return HistGradientBoostingRegressor(loss='squared_error', random_state=int(seed) % (2 ** 31 - 1), **cfg)


def _fit_pred(branch, Xtr, ytr, dtr, Xte, seed):
    cfg = CONFIGS[branch.replace('_bag', '')]
    if not branch.endswith('_bag'):
        m = _make(cfg, seed)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    rng = np.random.RandomState(int(seed) % (2 ** 31 - 1))
    udays = np.unique(dtr)
    preds = np.zeros(Xte.shape[0], dtype=float)
    for b in range(N_BAGS):
        pick = rng.choice(udays, size=len(udays), replace=True)
        cnt = collections.Counter(pick.tolist())
        w = np.array([cnt.get(float(d), 0) for d in dtr], dtype=float)
        mask = w > 0
        m = _make(cfg, int(seed) + 7919 * (b + 1))
        m.fit(Xtr[mask], ytr[mask], sample_weight=w[mask])
        preds += m.predict(Xte)
    return preds / float(N_BAGS)


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
    findings = {'branches': BRANCHES, 'n_bags': N_BAGS, 'configs': CONFIGS, 'folds': [],
                'fold_rmse': {b: [] for b in BRANCHES}, 'fold_date_mae': {b: [] for b in BRANCHES},
                'errors': {}, 'fallback': None,
                'note': 'same two forward 61-day training holdouts as idea-ae78bf; candidates: baseline HGB, slower/more regularized HGB, and date-bootstrap bags (6 bags, whole-date resampling via sample_weight multiplicity) of each; squared error throughout; selection by mean forward RMSE, ties by branch order; evaluation covariates used only for final prediction; no evaluation labels'}
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
                p = _fit_pred(b, X[tr], y[tr], days[tr], X[te], seed)
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
    selected = order[0] if np.isfinite(mean_rmse[order[0]]) else 'base'
    findings['mean_forward_rmse'] = {b: (mean_rmse[b] if np.isfinite(mean_rmse[b]) else None) for b in BRANCHES}
    findings['mean_forward_date_mae'] = {b: (mean_mae[b] if np.isfinite(mean_mae[b]) else None) for b in BRANCHES}
    findings['selected_branch'] = selected
    findings['best_by_date_mae'] = min(BRANCHES, key=lambda b: (mean_mae[b], BRANCHES.index(b)))
    findings['train_rows'] = int(X.shape[0])
    findings['train_day_range'] = [float(np.min(days)), dmax]
    findings['eval_day_range'] = [float(np.min(Xe[:, di])), float(np.max(Xe[:, di]))]
    final_preds = {}
    for b in BRANCHES:
        try:
            fp = np.asarray(_fit_pred(b, X, y, days, Xe, seed), dtype=float)
            fp = np.where(np.isfinite(fp), fp, np.median(y))
            final_preds[b] = np.maximum(fp, 0.0)
        except Exception as exc:
            findings['errors']['final_' + b] = type(exc).__name__
    if selected in final_preds:
        pred = final_preds[selected]
    elif 'base' in final_preds:
        findings['fallback'] = 'base_after_final_fit_failure'
        pred = final_preds['base']
    else:
        findings['fallback'] = 'train_median_after_all_final_fits_failed'
        pred = np.full(Xe.shape[0], float(np.median(y)))
    findings['final_pred_means'] = {b: float(np.mean(v)) for b, v in final_preds.items()}
    if 'base' in final_preds:
        findings['final_rmse_vs_base'] = {b: _rmse(v, final_preds['base']) for b, v in final_preds.items()}
    pred = np.asarray(pred, dtype=float)
    bad = ~np.isfinite(pred)
    findings['nonfinite_replaced'] = int(bad.sum())
    if bad.any():
        pred[bad] = float(np.median(y))
    pred = np.maximum(pred, 0.0)
    findings['pred_summary'] = {'mean': float(np.mean(pred)), 'max': float(np.max(pred)), 'min': float(np.min(pred))}
    return {'prediction': pred.tolist(), 'findings': json.loads(json.dumps(findings))}
