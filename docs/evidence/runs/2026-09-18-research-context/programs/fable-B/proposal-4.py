import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def _model():
    return HistGradientBoostingRegressor(loss='poisson', max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1, random_state=1729)

def _metrics(p, y):
    e = p - y
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))), 'bias': float(np.mean(e)), 'pred_mean': float(np.mean(p))}

def _growth_rate(day, y):
    ud, inv = np.unique(day, return_inverse=True)
    tot = np.bincount(inv, weights=y)
    cnt = np.bincount(inv).astype(float)
    mean_h = tot / np.maximum(cnt, 1.0)
    idx = {float(d): i for i, d in enumerate(ud)}
    logs = []
    for i, d in enumerate(ud):
        j = idx.get(float(d) - 365.0)
        if j is None or cnt[i] < 20 or cnt[j] < 20 or mean_h[i] <= 0 or mean_h[j] <= 0:
            continue
        logs.append(float(np.log(mean_h[i] / mean_h[j])))
    if len(logs) < 20:
        return None, len(logs)
    return float(np.median(logs)) / 365.0, len(logs)

def _fit_predict(Xtr, ytr, Xte, ed, b, ref_day):
    ltr = np.exp(b * (Xtr[:, ed] - ref_day))
    lte = np.exp(b * (Xte[:, ed] - ref_day))
    m = _model()
    m.fit(Xtr, ytr / ltr)
    p = np.clip(m.predict(Xte), 0.0, None) * lte
    return p, int(m.n_iter_)

def fit_predict(train, inputs, seed):
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xe = np.asarray(inputs['X'], dtype=float)
    names = list(train['feature_names'])
    ed = names.index('elapsed_day')
    day = X[:, ed]
    ud = np.unique(day)
    damps = [0.0, 0.5, 1.0]
    findings = {'fallback_used': False, 'design': 'poisson HGB (parent hparams, all features) fitted on y/L(day), L=exp(damp*b*(day-last_fit_day)); b = median log year-over-year ratio of daily mean hourly cnt / 365, estimated from the fitting window only; damp selected by mean RMSE over 3 chronological folds with 61-day horizon', 'horizon_days': 61, 'damps': damps}
    internal = {}
    for lo in [ud[-91], ud[-61], ud[-31]]:
        hi = lo + 61.0
        tr = day < lo
        ho = (day >= lo) & (day < hi)
        b, npairs = _growth_rate(day[tr], y[tr])
        fold = {'train_rows': int(tr.sum()), 'holdout_rows': int(ho.sum()), 'holdout_mean_y': float(y[ho].mean()), 'yoy_pairs': int(npairs), 'b_per_day': b, 'results': {}}
        for dmp in damps:
            key = 'damp_%.1f' % dmp
            if b is None and dmp > 0:
                fold['results'][key] = {'error': 'no_growth_estimate'}
                continue
            try:
                p, nit = _fit_predict(X[tr], y[tr], X[ho], ed, (b or 0.0) * dmp, float(day[tr].max()))
                r = _metrics(p, y[ho])
                r['n_iter'] = nit
            except Exception as exc:
                r = {'error': type(exc).__name__}
            fold['results'][key] = r
        internal['cut_%d' % int(lo)] = fold
    findings['internal'] = internal
    summary = {}
    for dmp in damps:
        key = 'damp_%.1f' % dmp
        vals = [internal[k]['results'][key] for k in internal if 'rmse' in internal[k]['results'][key]]
        if len(vals) == len(internal):
            summary[key] = {'mean_rmse': float(np.mean([v['rmse'] for v in vals])), 'mean_bias': float(np.mean([v['bias'] for v in vals])), 'mean_mae': float(np.mean([v['mae'] for v in vals]))}
    findings['internal_summary'] = summary
    if summary:
        sel_key = min(summary, key=lambda k: (summary[k]['mean_rmse'], k))
        sel = float(sel_key.split('_')[1])
    else:
        sel_key, sel = 'damp_0.0_default', 0.0
        findings['selection_note'] = 'no complete internal comparison; default damp 0'
    findings['selected'] = sel_key
    b_full, npairs = _growth_rate(day, y)
    findings['full_yoy_pairs'] = int(npairs)
    findings['full_b_per_day'] = b_full
    if b_full is None:
        b_full = 0.0
        findings['fallback_used'] = True
        findings['fallback_reason'] = 'no_growth_estimate_full'
    ref = float(ud[-1])
    findings['implied_factor_at_eval_mid'] = float(np.exp(sel * b_full * (float(np.median(Xe[:, ed])) - ref)))
    findings['implied_annual_factor_undamped'] = float(np.exp(b_full * 365.0))
    try:
        pred, nit = _fit_predict(X, y, Xe, ed, sel * b_full, ref)
        findings['final_n_iter'] = nit
        findings['submitted'] = 'poisson_trend_normalized_' + sel_key
    except Exception as exc:
        findings['fallback_used'] = True
        findings['fallback_reason'] = type(exc).__name__
        findings['submitted'] = 'poisson_plain_fallback'
        m = _model()
        m.fit(X, y)
        pred = np.clip(m.predict(Xe), 0.0, None)
    pred = np.asarray(pred, dtype=float)
    bad = ~np.isfinite(pred)
    if bad.any():
        findings['fallback_used'] = True
        findings['nonfinite_replaced'] = int(bad.sum())
        pred[bad] = float(np.median(y))
    findings['pred_mean'] = float(pred.mean())
    findings['train_last60_mean'] = float(y[day >= ud[-1] - 59].mean())
    findings['eval_day_range'] = [float(Xe[:, ed].min()), float(Xe[:, ed].max())]
    return {'prediction': pred.tolist(), 'findings': findings}
