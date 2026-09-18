import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def _hgb(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False, random_state=int(seed) % 2147483647)

def _daily_level(X, y, day, cols, seed, nblocks=5):
    ud = np.unique(day)
    ly = np.log1p(np.maximum(y, 0.0))
    res = np.zeros(len(y))
    for b in np.array_split(ud, nblocks):
        m = np.isin(day, b)
        if m.sum() == 0 or (~m).sum() == 0:
            continue
        mdl = _hgb(seed)
        mdl.fit(X[~m][:, cols], ly[~m])
        res[m] = ly[m] - mdl.predict(X[m][:, cols])
    rr = np.array([res[day == d].mean() for d in ud])
    return ud, rr

def _offset(dd, rr, mode, eday):
    if mode == 'none':
        return np.zeros(len(eday)), {}
    if mode == 'const60':
        w = dd >= dd[-1] - 59
        c = float(rr[w].mean())
        return np.full(len(eday), c), {'const': c, 'n_days': int(w.sum())}
    if mode == 'trend_all':
        a, b = np.polyfit(dd, rr, 1)
        return a * eday + b, {'slope_per_day': float(a), 'intercept': float(b), 'n_days': int(len(dd))}
    if mode == 'trend180':
        w = dd >= dd[-1] - 179
        a, b = np.polyfit(dd[w], rr[w], 1)
        return a * eday + b, {'slope_per_day': float(a), 'intercept': float(b), 'n_days': int(w.sum())}
    raise ValueError(mode)

def _predict_level(X, y, day, Xe, eday, cols, mode, seed):
    ly = np.log1p(np.maximum(y, 0.0))
    dd, rr = _daily_level(X, y, day, cols, seed)
    off, info = _offset(dd, rr, mode, eday)
    mdl = _hgb(seed)
    mdl.fit(X[:, cols], ly)
    lp = mdl.predict(Xe[:, cols]) + off
    p = np.expm1(np.minimum(lp, 12.0))
    info['mean_offset'] = float(np.mean(off))
    return np.clip(p, 0.0, None), info

def _predict_base(X, y, Xe, seed):
    mdl = _hgb(seed)
    mdl.fit(X, y)
    return np.clip(mdl.predict(Xe), 0.0, None)

def _metrics(p, y):
    e = p - y
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))), 'bias': float(np.mean(e))}

def fit_predict(train, inputs, seed):
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xe = np.asarray(inputs['X'], dtype=float)
    names = list(train['feature_names'])
    ed = names.index('elapsed_day')
    yi = names.index('yr')
    cols = [i for i in range(X.shape[1]) if i not in (ed, yi)]
    day = X[:, ed]
    eday = Xe[:, ed]
    findings = {'fallback_used': False, 'shape_features_exclude': ['elapsed_day', 'yr'], 'level_model': 'cross_fitted_daily_mean_log1p_residual_5_date_blocks', 'target_for_level_variants': 'log1p'}
    ud = np.unique(day)
    modes = ['none', 'const60', 'trend_all', 'trend180']
    folds = [(ud[-60], ud[-30]), (ud[-30], ud[-1] + 1.0)]
    internal = {}
    for fi, (lo, hi) in enumerate(folds):
        tr = day < lo
        ho = (day >= lo) & (day < hi)
        fold_res = {}
        pb = _predict_base(X[tr], y[tr], X[ho], seed)
        fold_res['baseline_all_raw'] = _metrics(pb, y[ho])
        for mode in modes:
            p, info = _predict_level(X[tr], y[tr], day[tr], X[ho], day[ho], cols, mode, seed)
            r = _metrics(p, y[ho])
            r.update(info)
            fold_res['level_' + mode] = r
        internal['fold%d' % fi] = {'cut_lo': float(lo), 'cut_hi': float(hi), 'holdout_rows': int(ho.sum()), 'train_rows': int(tr.sum()), 'holdout_mean_y': float(y[ho].mean()), 'results': fold_res}
    cands = ['baseline_all_raw'] + ['level_' + m for m in modes]
    avg = {c: float(np.mean([internal[f]['results'][c]['rmse'] for f in internal])) for c in cands}
    best = min(avg, key=avg.get)
    findings['internal'] = internal
    findings['avg_internal_rmse'] = avg
    findings['selected'] = best
    try:
        dd, rr = _daily_level(X, y, day, cols, seed)
        a, b = np.polyfit(dd, rr, 1)
        w60 = dd >= dd[-1] - 59
        f60 = dd <= dd[0] + 59
        findings['full_train_level'] = {'slope_per_day_all': float(a), 'mean_first60': float(rr[f60].mean()), 'mean_last60': float(rr[w60].mean()), 'std_daily': float(rr.std())}
    except Exception as exc:
        findings['level_diag_error'] = type(exc).__name__
    if best == 'baseline_all_raw':
        pred = _predict_base(X, y, Xe, seed)
        findings['final_level_info'] = {}
    else:
        mode = best[len('level_'):]
        pred, info = _predict_level(X, y, day, Xe, eday, cols, mode, seed)
        findings['final_level_info'] = info
    pred = np.asarray(pred, dtype=float)
    bad = ~np.isfinite(pred)
    if bad.any():
        findings['fallback_used'] = True
        findings['nonfinite_replaced'] = int(bad.sum())
        pred[bad] = float(np.median(y))
    findings['pred_mean'] = float(pred.mean())
    findings['train_mean'] = float(y.mean())
    findings['train_last60_mean'] = float(y[day >= ud[-1] - 59].mean())
    findings['eval_day_range'] = [float(eday.min()), float(eday.max())]
    findings['train_day_range'] = [float(day.min()), float(day.max())]
    return {'prediction': pred.tolist(), 'findings': findings}
