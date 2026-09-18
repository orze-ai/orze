import numpy as np
import json
from sklearn.ensemble import HistGradientBoostingRegressor

BRANCHES = ['base', 'yoy_scaled', 'blend']
LAG = 364
HALF = 14
RATIO_WINDOW = 56
RATIO_CLIP = (0.5, 4.0)
DROP_FOR_SCALED = ['yr', 'elapsed_day']


def _make(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1, loss='squared_error', random_state=seed)


def _rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _date_mae(p, y, days):
    vals = [float(np.mean(np.abs(p[days == d] - y[days == d]))) for d in np.unique(days)]
    return float(np.mean(vals)) if vals else float('nan')


def _key(d):
    return float(round(float(d)))


def _level_model(days, y):
    raw = {}
    for d in np.unique(days):
        raw[_key(d)] = float(np.mean(y[days == d]))
    keys = np.array(sorted(raw.keys()), dtype=float)
    vals = np.array([raw[k] for k in keys], dtype=float)
    smooth = {}
    for k in keys:
        w = (keys >= k - HALF) & (keys <= k + HALF)
        smooth[float(k)] = float(np.mean(vals[w]))
    last_day = float(keys[-1])
    ratios = []
    for k in keys[keys > last_day - RATIO_WINDOW]:
        a = smooth.get(_key(k - LAG))
        if a is not None and a > 0:
            ratios.append(smooth[float(k)] / a)
    ratio = None
    clipped = False
    if ratios:
        ratio = float(np.exp(np.mean(np.log(ratios))))
        if ratio < RATIO_CLIP[0] or ratio > RATIO_CLIP[1]:
            clipped = True
            ratio = float(min(max(ratio, RATIO_CLIP[0]), RATIO_CLIP[1]))
    return {'smooth': smooth, 'last_day': last_day, 'last_level': smooth[last_day],
            'ratio': ratio, 'ratio_clipped': bool(clipped), 'n_ratio_days': int(len(ratios))}


def _train_level(days, lm):
    return np.array([lm['smooth'][_key(d)] for d in days], dtype=float)


def _eval_level(days, lm):
    out = np.empty(len(days), dtype=float)
    persist = np.zeros(len(days), dtype=bool)
    for i, d in enumerate(days):
        a = lm['smooth'].get(_key(d - LAG)) if lm['ratio'] is not None else None
        if a is not None and a > 0:
            out[i] = a * lm['ratio']
        else:
            out[i] = lm['last_level']
            persist[i] = True
    return out, persist


def _branch_preds(Xtr, ytr, Xte, cols, di, seed):
    info = {}
    mb = _make(seed)
    mb.fit(Xtr, ytr)
    pb = mb.predict(Xte)
    lm = _level_model(Xtr[:, di], ytr)
    ltr = _train_level(Xtr[:, di], lm)
    lte, persist = _eval_level(Xte[:, di], lm)
    z = ytr / np.maximum(ltr, 1e-6)
    ms = _make(seed)
    ms.fit(Xtr[:, cols], z)
    ps = np.maximum(ms.predict(Xte[:, cols]), 0.0) * lte
    info['yoy_ratio'] = lm['ratio']
    info['ratio_clipped'] = lm['ratio_clipped']
    info['n_ratio_days'] = lm['n_ratio_days']
    info['last_train_level'] = float(lm['last_level'])
    info['persistence_frac'] = float(np.mean(persist))
    info['eval_level_mean'] = float(np.mean(lte))
    info['train_level_mean'] = float(np.mean(ltr))
    info['z_mean'] = float(np.mean(z))
    return {'base': pb, 'yoy_scaled': ps, 'blend': 0.5 * (pb + ps)}, info


def _mean_over(lst, idx):
    vals = [lst[i] for i in idx]
    return float(np.mean(vals)) if vals else float('inf')


def fit_predict(train, inputs, seed):
    fn = list(train['feature_names'])
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xe = np.asarray(inputs['X'], dtype=float)
    di = fn.index('elapsed_day')
    cols = [i for i, n in enumerate(fn) if n not in DROP_FOR_SCALED]
    days = X[:, di]
    dmax = float(np.max(days))
    findings = {'branches': BRANCHES, 'folds': [], 'fold_rmse': {b: [] for b in BRANCHES},
                'fold_date_mae': {b: [] for b in BRANCHES}, 'errors': {}, 'fallback': None,
                'scaled_features': [fn[i] for i in cols],
                'note': 'yoy_scaled: HGB (baseline hyperparameters, squared error) fit on cnt divided by a +/-14-day smoothed mean hourly count per training date; evaluation-date level = smoothed level at elapsed_day-364 times geometric-mean YoY ratio over trailing 56 training days (clipped to [0.5,4]); persistence of last training level when no analog; blend = 0.5*(base+yoy_scaled); selection by mean forward RMSE over training folds with analog coverage >= 50 percent (all folds if none); only evaluation elapsed_day is used for analog lookup; no evaluation labels anywhere'}
    fold_records = []
    for k in range(2):
        hi = dmax - 61.0 * k
        lo = hi - 61.0
        te = (days > lo) & (days <= hi)
        tr = days <= lo
        if te.sum() < 100 or tr.sum() < 500:
            findings['folds'].append({'fold': k + 1, 'skipped': True})
            continue
        rec = {'fold': k + 1, 'holdout_day_range': [float(lo), float(hi)],
               'holdout_rows': int(te.sum()), 'train_rows': int(tr.sum())}
        try:
            preds, info = _branch_preds(X[tr], y[tr], X[te], cols, di, seed)
            rec['level_info'] = info
            rec['analog_supported'] = bool(info['persistence_frac'] < 0.5)
            for b in BRANCHES:
                p = np.asarray(preds[b], dtype=float)
                p = np.where(np.isfinite(p), p, np.median(y[tr]))
                p = np.maximum(p, 0.0)
                findings['fold_rmse'][b].append(_rmse(p, y[te]))
                findings['fold_date_mae'][b].append(_date_mae(p, y[te], days[te]))
        except Exception as exc:
            findings['errors']['fold' + str(k + 1)] = type(exc).__name__
            rec['analog_supported'] = False
            for b in BRANCHES:
                findings['fold_rmse'][b].append(float('inf'))
                findings['fold_date_mae'][b].append(float('inf'))
        findings['folds'].append(rec)
        fold_records.append(rec)
    sel_idx = [i for i, r in enumerate(fold_records) if r.get('analog_supported')]
    all_idx = list(range(len(fold_records)))
    if not sel_idx:
        sel_idx = all_idx
    findings['selection_fold_numbers'] = [fold_records[i]['fold'] for i in sel_idx]
    mean_sel = {b: _mean_over(findings['fold_rmse'][b], sel_idx) for b in BRANCHES}
    mean_all = {b: _mean_over(findings['fold_rmse'][b], all_idx) for b in BRANCHES}
    mae_sel = {b: _mean_over(findings['fold_date_mae'][b], sel_idx) for b in BRANCHES}
    order = sorted(BRANCHES, key=lambda b: (mean_sel[b], BRANCHES.index(b)))
    selected = order[0] if np.isfinite(mean_sel[order[0]]) else 'base'
    order_all = sorted(BRANCHES, key=lambda b: (mean_all[b], BRANCHES.index(b)))
    findings['selection_rmse'] = {b: (mean_sel[b] if np.isfinite(mean_sel[b]) else None) for b in BRANCHES}
    findings['all_fold_rmse_mean'] = {b: (mean_all[b] if np.isfinite(mean_all[b]) else None) for b in BRANCHES}
    findings['selection_date_mae'] = {b: (mae_sel[b] if np.isfinite(mae_sel[b]) else None) for b in BRANCHES}
    findings['selected_branch'] = selected
    findings['would_select_all_folds'] = order_all[0] if np.isfinite(mean_all[order_all[0]]) else 'base'
    findings['best_by_date_mae'] = min(BRANCHES, key=lambda b: (mae_sel[b], BRANCHES.index(b)))
    findings['train_rows'] = int(X.shape[0])
    findings['train_day_range'] = [float(np.min(days)), dmax]
    findings['eval_day_range'] = [float(np.min(Xe[:, di])), float(np.max(Xe[:, di]))]
    try:
        preds, info = _branch_preds(X, y, Xe, cols, di, seed)
        findings['final_level_info'] = info
        pred = np.asarray(preds[selected], dtype=float)
        findings['final_branch_pred_means'] = {b: float(np.mean(preds[b])) for b in BRANCHES}
    except Exception as exc:
        findings['errors']['final_' + selected] = type(exc).__name__
        findings['fallback'] = 'base_hgb_after_final_fit_failure'
        m = _make(seed)
        m.fit(X, y)
        pred = np.asarray(m.predict(Xe), dtype=float)
    bad = ~np.isfinite(pred)
    findings['nonfinite_replaced'] = int(bad.sum())
    if bad.any():
        pred[bad] = float(np.median(y))
    pred = np.maximum(pred, 0.0)
    findings['pred_summary'] = {'mean': float(np.mean(pred)), 'max': float(np.max(pred)), 'min': float(np.min(pred))}
    return {'prediction': pred.tolist(), 'findings': json.loads(json.dumps(findings))}
