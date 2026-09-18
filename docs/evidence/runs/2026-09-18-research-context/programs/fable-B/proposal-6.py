import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HP = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1, random_state=1729)

def _model(loss):
    return HistGradientBoostingRegressor(loss=loss, **HP)

def _r(x):
    return float(round(float(x), 3))

def _metrics(p, y):
    e = p - y
    return {'rmse': _r(np.sqrt(np.mean(e ** 2))), 'mae': _r(np.mean(np.abs(e))), 'bias': _r(np.mean(e)), 'pred_mean': _r(np.mean(p))}

def _fit_base(spec, Xtr, ytr, Xte):
    loss, drop = spec
    cols = [i for i in range(Xtr.shape[1]) if i not in drop]
    m = _model(loss)
    m.fit(Xtr[:, cols], ytr)
    return np.clip(m.predict(Xte[:, cols]), 0.0, None), int(m.n_iter_)

def fit_predict(train, inputs, seed):
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xe = np.asarray(inputs['X'], dtype=float)
    names = list(train['feature_names'])
    ed = names.index('elapsed_day')
    day = X[:, ed]
    ud = np.unique(day)
    drop_ms = tuple(names.index(n) for n in ['mnth', 'season'] if n in names)
    base = {'poisson_all': ('poisson', ()), 'poisson_no_month_season': ('poisson', drop_ms), 'squared_all': ('squared_error', ())}
    combos = [('poisson_all', ['poisson_all']), ('poisson_no_month_season', ['poisson_no_month_season']), ('squared_all', ['squared_all']), ('avg_poisson_all_squared', ['poisson_all', 'squared_all']), ('avg_poisson_all_noms', ['poisson_all', 'poisson_no_month_season']), ('avg_all_three', ['poisson_all', 'poisson_no_month_season', 'squared_all'])]
    order = {k: i for i, (k, _) in enumerate(combos)}
    findings = {'fallback_used': False, 'design': 'HGB variants with idea-a0bc19 hparams (max_iter=250,lr=0.08,leaves=31,l2=1,seed=1729): poisson all features, poisson without mnth/season, squared_error all features, and their simple averages; branch selected by mean RMSE over 3 chronological folds with 61-day horizon, ties toward poisson_all (identical to a0bc19 submission)', 'dropped_for_no_month_season': [names[i] for i in drop_ms], 'horizon_days': 61, 'candidates': [k for k, _ in combos]}
    internal = {}
    for lo in [ud[-91], ud[-61], ud[-31]]:
        hi = lo + 61.0
        tr = day < lo
        ho = (day >= lo) & (day < hi)
        preds = {}
        errs = {}
        for bname, spec in base.items():
            try:
                p, nit = _fit_base(spec, X[tr], y[tr], X[ho])
                preds[bname] = p
            except Exception as exc:
                errs[bname] = type(exc).__name__
        res = {}
        for cname, parts in combos:
            if all(b in preds for b in parts):
                res[cname] = _metrics(np.mean([preds[b] for b in parts], axis=0), y[ho])
            else:
                res[cname] = {'error': 'missing_base'}
        fold = {'train_rows': int(tr.sum()), 'holdout_rows': int(ho.sum()), 'holdout_mean_y': _r(y[ho].mean()), 'results': res}
        if errs:
            fold['base_errors'] = errs
        internal['cut_%d' % int(lo)] = fold
    findings['internal'] = internal
    summary = {}
    for cname, _ in combos:
        vals = [internal[k]['results'][cname] for k in internal if 'rmse' in internal[k]['results'][cname]]
        if len(vals) == len(internal):
            summary[cname] = {'mean_rmse': _r(np.mean([v['rmse'] for v in vals])), 'mean_mae': _r(np.mean([v['mae'] for v in vals])), 'mean_bias': _r(np.mean([v['bias'] for v in vals]))}
    findings['internal_summary'] = summary
    if summary:
        sel = min(summary, key=lambda k: (summary[k]['mean_rmse'], order[k]))
    else:
        sel = 'poisson_all'
        findings['selection_note'] = 'no complete internal comparison; default poisson_all'
    findings['selected'] = sel
    parts = dict(combos)[sel]
    findings['selected_components'] = parts
    findings['identical_to_a0bc19_submission'] = bool(sel == 'poisson_all')
    try:
        outs = []
        iters = {}
        for b in parts:
            p, nit = _fit_base(base[b], X, y, Xe)
            outs.append(p)
            iters[b] = nit
        pred = np.mean(outs, axis=0)
        findings['final_n_iter'] = iters
        findings['submitted'] = sel
    except Exception as exc:
        findings['fallback_used'] = True
        findings['fallback_reason'] = type(exc).__name__
        try:
            pred, nit = _fit_base(base['poisson_all'], X, y, Xe)
            findings['submitted'] = 'poisson_all_fallback'
        except Exception as exc2:
            findings['fallback_reason2'] = type(exc2).__name__
            pred = np.full(Xe.shape[0], float(np.median(y)))
            findings['submitted'] = 'train_median_fallback'
    pred = np.asarray(pred, dtype=float)
    bad = ~np.isfinite(pred)
    if bad.any():
        findings['fallback_used'] = True
        findings['nonfinite_replaced'] = int(bad.sum())
        pred[bad] = float(np.median(y))
    findings['pred_mean'] = _r(pred.mean())
    findings['train_mean'] = _r(y.mean())
    findings['train_last60_mean'] = _r(y[day >= ud[-1] - 59].mean())
    findings['eval_day_range'] = [float(Xe[:, ed].min()), float(Xe[:, ed].max())]
    return {'prediction': pred.tolist(), 'findings': findings}
