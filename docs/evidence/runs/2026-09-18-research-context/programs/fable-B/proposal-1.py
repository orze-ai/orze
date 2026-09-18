import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def _make(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False, random_state=int(seed) % 2147483647)

def _f(name, v):
    if name == 'log1p':
        return np.log1p(np.maximum(v, 0.0))
    if name == 'sqrt':
        return np.sqrt(np.maximum(v, 0.0))
    return v

def _finv(name, v):
    if name == 'log1p':
        return np.expm1(np.minimum(v, 12.0))
    if name == 'sqrt':
        return np.square(np.maximum(v, 0.0))
    return v

def fit_predict(train, inputs, seed):
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xe = np.asarray(inputs['X'], dtype=float)
    names = list(train['feature_names'])
    findings = {'fallback_used': False, 'early_stopping': False}
    ed = names.index('elapsed_day')
    day = X[:, ed]
    udays = np.unique(day)
    ncut = 61 if len(udays) > 122 else max(1, len(udays) // 4)
    cut = udays[-ncut]
    tr = day < cut
    ho = ~tr
    findings['internal_holdout'] = {'days': int(ncut), 'rows': int(ho.sum()), 'cut_elapsed_day': float(cut), 'train_rows': int(tr.sum())}
    fsets = {'all': list(range(X.shape[1])), 'no_elapsed_day': [i for i in range(X.shape[1]) if i != ed]}
    results = {}
    for t in ['raw', 'log1p', 'sqrt']:
        for fs, cols in fsets.items():
            m = _make(seed)
            m.fit(X[tr][:, cols], _f(t, y[tr]))
            p = np.clip(_finv(t, m.predict(X[ho][:, cols])), 0.0, None)
            e = p - y[ho]
            results[t + '|' + fs] = {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))), 'bias': float(np.mean(e))}
    best = min(results, key=lambda k: results[k]['rmse'])
    t, fs = best.split('|')
    cols = fsets[fs]
    findings['internal_results'] = results
    findings['selected'] = {'transform': t, 'features': fs}
    try:
        yi = names.index('yr')
        mi = names.index('mnth')
        lv = {}
        for mo in np.unique(X[:, mi]):
            a = y[(X[:, mi] == mo) & (X[:, yi] == 0)]
            b = y[(X[:, mi] == mo) & (X[:, yi] == 1)]
            if len(a) and len(b):
                lv[str(int(mo))] = {'mean_2011': float(a.mean()), 'mean_2012': float(b.mean()), 'ratio': float(b.mean() / max(a.mean(), 1e-9))}
        findings['year_level_by_month'] = lv
    except Exception as exc:
        findings['level_diag_error'] = type(exc).__name__
    m = _make(seed)
    m.fit(X[:, cols], _f(t, y))
    pred = np.clip(_finv(t, m.predict(Xe[:, cols])), 0.0, None)
    bad = ~np.isfinite(pred)
    if bad.any():
        findings['fallback_used'] = True
        findings['nonfinite_replaced'] = int(bad.sum())
        pred[bad] = float(np.median(y))
    findings['pred_mean'] = float(pred.mean())
    findings['train_mean'] = float(y.mean())
    return {'prediction': pred.tolist(), 'findings': findings}
