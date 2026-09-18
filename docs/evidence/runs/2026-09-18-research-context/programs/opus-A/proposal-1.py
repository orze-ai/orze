import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
        'weathersit', 'temp', 'atemp', 'hum', 'windspeed']


def _features(X, feature_names, use_elapsed):
    A = np.asarray(X, dtype=float)
    ix = {}
    for i, nm in enumerate(feature_names):
        ix[nm] = i
    n = A.shape[0]
    z = np.zeros(n)

    def col(name):
        if name in ix:
            return A[:, ix[name]]
        return z

    cols = [col(b) for b in BASE]
    if use_elapsed and ('elapsed_day' in ix):
        cols.append(col('elapsed_day'))
    hr = col('hr')
    wd = col('workingday')
    wk = col('weekday')
    tp = col('temp')
    hu = col('hum')
    ws = col('windspeed')
    cols.append(np.sin(2.0 * np.pi * hr / 24.0))
    cols.append(np.cos(2.0 * np.pi * hr / 24.0))
    cols.append(np.sin(4.0 * np.pi * hr / 24.0))
    cols.append(np.cos(4.0 * np.pi * hr / 24.0))
    cols.append(hr + 24.0 * wd)
    cols.append(hr + 24.0 * wk)
    cols.append(tp * hr)
    cols.append(hu * hr)
    cols.append(tp * wd)
    cols.append(ws * tp)
    return np.column_stack(cols)


def _fit_branch(F, y, idx, br, rs):
    model = HistGradientBoostingRegressor(max_iter=br['it'], learning_rate=br['lr'],
                                          max_leaf_nodes=br['lv'],
                                          l2_regularization=br['l2'],
                                          min_samples_leaf=br['ml'],
                                          early_stopping=False, random_state=rs)
    yt = y[idx]
    if br['target'] == 'log':
        yt = np.log1p(yt)
    model.fit(F[idx], yt)
    return model


def _predict_branch(model, F, br):
    p = model.predict(F)
    if br['target'] == 'log':
        p = np.expm1(np.clip(p, -20.0, 20.0))
    return np.clip(p, 0.0, None)


def _run(train, inputs, seed, findings):
    try:
        rs = int(seed)
    except Exception:
        rs = 1729
    fn = list(train['feature_names'])
    y = np.asarray(train['y'], dtype=float)
    groups = list(train['groups'])
    A = np.asarray(train['X'], dtype=float)
    ixe = fn.index('elapsed_day') if ('elapsed_day' in fn) else None
    key = {}
    for i, g in enumerate(groups):
        k = float(A[i, ixe]) if ixe is not None else float(i)
        if (g not in key) or (k < key[g]):
            key[g] = k
    order = sorted(key.keys(), key=lambda g: key[g])
    nd = len(order)
    H = 61
    folds = []
    for start in (nd - H, nd - 2 * H):
        if start >= 120:
            cut = key[order[start]]
            hold = set(order[start:start + H])
            tr_idx = np.array([i for i, g in enumerate(groups) if key[g] < cut], dtype=int)
            ho_idx = np.array([i for i, g in enumerate(groups) if g in hold], dtype=int)
            if tr_idx.size >= 500 and ho_idx.size >= 100:
                folds.append((tr_idx, ho_idx))
    feats = {True: _features(train['X'], fn, True),
             False: _features(train['X'], fn, False)}
    cfgs = [('ref', 250, 0.08, 31, 1.0, 20), ('deep', 600, 0.04, 63, 1.0, 15)]
    branches = []
    for ue in (True, False):
        for tgt in ('raw', 'log'):
            for c in cfgs:
                nm = '%s|%s|%s' % ('elapsed' if ue else 'noelapsed', tgt, c[0])
                branches.append({'name': nm, 'elapsed': ue, 'target': tgt,
                                 'it': c[1], 'lr': c[2], 'lv': c[3],
                                 'l2': c[4], 'ml': c[5]})
    scores = []
    for br in branches:
        F = feats[br['elapsed']]
        rl = []
        for tr_idx, ho_idx in folds:
            m = _fit_branch(F, y, tr_idx, br, rs)
            p = _predict_branch(m, F[ho_idx], br)
            rl.append(float(np.sqrt(np.mean((p - y[ho_idx]) ** 2))))
        if rl:
            mv = float(np.mean(rl))
        else:
            mv = None
        scores.append({'branch': br['name'],
                       'fold_rmse': [round(v, 3) for v in rl],
                       'mean_rmse': (round(mv, 3) if mv is not None else None)})
    if folds:
        vals = []
        for s in scores:
            vals.append(s['mean_rmse'] if s['mean_rmse'] is not None else 1e18)
        best_i = int(np.argmin(np.asarray(vals, dtype=float)))
    else:
        best_i = 0
        findings['no_internal_folds'] = True
    best = branches[best_i]
    Ftr = feats[best['elapsed']]
    all_idx = np.arange(y.shape[0], dtype=int)
    model = _fit_branch(Ftr, y, all_idx, best, rs)
    Fte = _features(inputs['X'], list(inputs['feature_names']), best['elapsed'])
    pred = _predict_branch(model, Fte, best)
    if (pred.shape[0] != Fte.shape[0]) or (not np.all(np.isfinite(pred))):
        raise ValueError('nonfinite_or_shape_mismatch')
    Ain = np.asarray(inputs['X'], dtype=float)
    ixe2 = None
    infn = list(inputs['feature_names'])
    if 'elapsed_day' in infn:
        ixe2 = infn.index('elapsed_day')
    findings['selected_branch'] = best['name']
    findings['branch_scores'] = scores
    findings['n_internal_folds'] = len(folds)
    findings['n_train_dates'] = nd
    findings['n_train_rows'] = int(y.shape[0])
    findings['n_pred_rows'] = int(pred.shape[0])
    findings['pred_mean'] = round(float(np.mean(pred)), 3)
    findings['pred_max'] = round(float(np.max(pred)), 3)
    findings['train_y_mean'] = round(float(np.mean(y)), 3)
    if (ixe is not None) and (ixe2 is not None):
        tmax = float(np.max(A[:, ixe]))
        imax = float(np.max(Ain[:, ixe2]))
        imin = float(np.min(Ain[:, ixe2]))
        findings['elapsed_train_max'] = tmax
        findings['elapsed_inputs_min'] = imin
        findings['elapsed_inputs_max'] = imax
        findings['inputs_beyond_train_elapsed_fraction'] = round(
            float(np.mean(Ain[:, ixe2] > tmax)), 4)
    findings['transductive_use'] = ('none_for_fitting_or_selection; unlabeled input '
                                    'covariates used only to predict and to report '
                                    'elapsed_day range diagnostics')
    findings['selection_rule'] = ('lowest mean RMSE over two strictly chronological '
                                  '61-date holdouts taken from training dates only')
    return {'prediction': [float(v) for v in pred], 'findings': findings}


def fit_predict(train, inputs, seed):
    findings = {'fallback_used': False}
    try:
        return _run(train, inputs, seed, findings)
    except Exception as exc:
        findings['fallback_used'] = True
        findings['fallback_reason'] = type(exc).__name__
        findings['fallback_model'] = ('HistGradientBoosting(250, lr 0.08, 31 leaves, '
                                      'l2 1, seed 1729) on raw supplied features; '
                                      'no branch selection was performed')
        Xtr = np.asarray(train['X'], dtype=float)
        ytr = np.asarray(train['y'], dtype=float)
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          random_state=1729)
        m.fit(Xtr, ytr)
        p = np.clip(m.predict(np.asarray(inputs['X'], dtype=float)), 0.0, None)
        p = np.nan_to_num(p, nan=float(np.mean(ytr)), posinf=float(np.max(ytr)), neginf=0.0)
        return {'prediction': [float(v) for v in p], 'findings': findings}
