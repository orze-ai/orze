import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def _model(loss):
    return HistGradientBoostingRegressor(loss=loss, max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1, random_state=1729)

def _metrics(p, y):
    e = p - y
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))), 'bias': float(np.mean(e)), 'pred_mean': float(np.mean(p))}

def fit_predict(train, inputs, seed):
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xe = np.asarray(inputs['X'], dtype=float)
    names = list(train['feature_names'])
    ed = names.index('elapsed_day')
    day = X[:, ed]
    ud = np.unique(day)
    findings = {'fallback_used': False, 'submitted': 'poisson_all_features', 'design': 'fixed submission; internal chronological folds are diagnostics only, not used for selection', 'shared_hparams': 'max_iter=250,lr=0.08,max_leaf_nodes=31,l2=1,random_state=1729,early_stopping=auto'}
    cands = {'squared_all': ('squared_error', None), 'poisson_all': ('poisson', None), 'poisson_no_elapsed': ('poisson', ed)}
    internal = {}
    for lo in [ud[-90], ud[-60], ud[-30]]:
        hi = lo + 30.0
        tr = day < lo
        ho = (day >= lo) & (day < hi)
        res = {}
        for name, (loss, drop) in cands.items():
            cols = [i for i in range(X.shape[1]) if i != drop]
            try:
                m = _model(loss)
                m.fit(X[tr][:, cols], y[tr])
                p = np.clip(m.predict(X[ho][:, cols]), 0.0, None)
                r = _metrics(p, y[ho])
                r['n_iter'] = int(m.n_iter_)
            except Exception as exc:
                r = {'error': type(exc).__name__}
            res[name] = r
        internal['cut_%d' % int(lo)] = {'train_rows': int(tr.sum()), 'holdout_rows': int(ho.sum()), 'holdout_mean_y': float(y[ho].mean()), 'results': res}
    findings['internal'] = internal
    summary = {}
    for name in cands:
        vals = [internal[k]['results'][name] for k in internal if 'rmse' in internal[k]['results'][name]]
        if vals:
            summary[name] = {'mean_rmse': float(np.mean([v['rmse'] for v in vals])), 'mean_bias': float(np.mean([v['bias'] for v in vals])), 'n_folds': len(vals)}
    findings['internal_summary'] = summary
    m11 = (day >= 0) & (day <= 120)
    m12 = (day >= 365) & (day <= 485)
    if m11.sum() > 0 and m12.sum() > 0:
        findings['jan_apr_level_ratio_2012_over_2011'] = float(y[m12].mean() / max(y[m11].mean(), 1e-9))
    try:
        m = _model('poisson')
        m.fit(X, y)
        pred = np.clip(m.predict(Xe), 0.0, None)
        findings['final_n_iter'] = int(m.n_iter_)
    except Exception as exc:
        findings['fallback_used'] = True
        findings['fallback_reason'] = type(exc).__name__
        findings['submitted'] = 'squared_error_fallback'
        m = _model('squared_error')
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
