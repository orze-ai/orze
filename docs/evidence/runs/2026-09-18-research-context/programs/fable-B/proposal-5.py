import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def _model():
    return HistGradientBoostingRegressor(loss='poisson', max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1, random_state=1729)

def _metrics(p, y):
    e = p - y
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))), 'bias': float(np.mean(e)), 'pred_mean': float(np.mean(p))}

def _weights(day, hl):
    if hl is None:
        return np.ones_like(day, dtype=float)
    w = np.power(0.5, (float(day.max()) - day) / float(hl))
    return w / w.mean()

def _ess(w):
    return float(w.sum() ** 2 / np.sum(w ** 2))

def _fit_predict(Xtr, ytr, Xte, day_tr, hl):
    m = _model()
    m.fit(Xtr, ytr, sample_weight=_weights(day_tr, hl))
    return np.clip(m.predict(Xte), 0.0, None), int(m.n_iter_)

def fit_predict(train, inputs, seed):
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xe = np.asarray(inputs['X'], dtype=float)
    names = list(train['feature_names'])
    ed = names.index('elapsed_day')
    day = X[:, ed]
    ud = np.unique(day)
    cands = [('hl_none', None), ('hl_365', 365.0), ('hl_180', 180.0), ('hl_90', 90.0)]
    order = {k: i for i, (k, _) in enumerate(cands)}
    findings = {'fallback_used': False, 'design': 'poisson HGB (idea-a0bc19 hparams, all features) with exponential recency sample weights w=0.5**((last_fit_day-day)/half_life), normalized to mean 1; half-life selected by mean RMSE over 3 chronological folds with 61-day horizon, ties toward hl_none; only sample weights vary', 'horizon_days': 61, 'candidates': [k for k, _ in cands]}
    internal = {}
    for lo in [ud[-91], ud[-61], ud[-31]]:
        hi = lo + 61.0
        tr = day < lo
        ho = (day >= lo) & (day < hi)
        fold = {'train_rows': int(tr.sum()), 'holdout_rows': int(ho.sum()), 'holdout_mean_y': float(y[ho].mean()), 'results': {}}
        for key, hl in cands:
            try:
                p, nit = _fit_predict(X[tr], y[tr], X[ho], day[tr], hl)
                r = _metrics(p, y[ho])
                r['n_iter'] = nit
                r['ess'] = _ess(_weights(day[tr], hl))
            except Exception as exc:
                r = {'error': type(exc).__name__}
            fold['results'][key] = r
        internal['cut_%d' % int(lo)] = fold
    findings['internal'] = internal
    summary = {}
    for key, hl in cands:
        vals = [internal[k]['results'][key] for k in internal if 'rmse' in internal[k]['results'][key]]
        if len(vals) == len(internal):
            summary[key] = {'mean_rmse': float(np.mean([v['rmse'] for v in vals])), 'mean_bias': float(np.mean([v['bias'] for v in vals])), 'mean_mae': float(np.mean([v['mae'] for v in vals]))}
    findings['internal_summary'] = summary
    if summary:
        sel_key = min(summary, key=lambda k: (summary[k]['mean_rmse'], order[k]))
    else:
        sel_key = 'hl_none'
        findings['selection_note'] = 'no complete internal comparison; default hl_none'
    sel_hl = dict(cands)[sel_key]
    findings['selected'] = sel_key
    findings['selected_half_life_days'] = sel_hl
    try:
        pred, nit = _fit_predict(X, y, Xe, day, sel_hl)
        findings['final_n_iter'] = nit
        findings['final_ess'] = _ess(_weights(day, sel_hl))
        findings['submitted'] = 'poisson_recency_' + sel_key
    except Exception as exc:
        findings['fallback_used'] = True
        findings['fallback_reason'] = type(exc).__name__
        findings['submitted'] = 'poisson_uniform_fallback'
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
    findings['train_mean'] = float(y.mean())
    findings['train_last60_mean'] = float(y[day >= ud[-1] - 59].mean())
    findings['eval_day_range'] = [float(Xe[:, ed].min()), float(Xe[:, ed].max())]
    return {'prediction': pred.tolist(), 'findings': findings}
