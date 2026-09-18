import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold


def log_index(names):
    out = []
    for i, n in enumerate(names):
        low = str(n).lower()
        if ('jitter' in low) or ('shimmer' in low) or ('nhr' in low):
            out.append(i)
    return out


def apply_log(Z, lidx):
    Z = Z.copy()
    for i in lidx:
        Z[:, i] = np.log1p(np.maximum(Z[:, i], 0.0))
    return Z


def col_subset(names, mode):
    demo = []
    voice = []
    for i, n in enumerate(names):
        low = str(n).lower()
        if low in ('age', 'sex', 'test_time'):
            demo.append(i)
        else:
            voice.append(i)
    if mode == 'all':
        return list(range(len(names)))
    if mode == 'voice':
        return voice
    if mode == 'demo':
        return [i for i in demo if str(names[i]).lower() != 'test_time']
    if mode == 'demo_time':
        return demo
    return list(range(len(names)))


def make_specs():
    specs = []
    specs.append({'name': 'median', 'kind': 'const', 'stat': 'median'})
    specs.append({'name': 'mean', 'kind': 'const', 'stat': 'mean'})
    specs.append({'name': 'lin_age_sex', 'kind': 'ridge', 'cols': 'demo', 'alpha': 0.001, 'log': False})
    specs.append({'name': 'lin_age_sex_time', 'kind': 'ridge', 'cols': 'demo_time', 'alpha': 0.001, 'log': False})
    for a in (10.0, 100.0, 1000.0):
        specs.append({'name': 'ridge_all_a%g' % a, 'kind': 'ridge', 'cols': 'all', 'alpha': a, 'log': False})
        specs.append({'name': 'ridge_all_log_a%g' % a, 'kind': 'ridge', 'cols': 'all', 'alpha': a, 'log': True})
    specs.append({'name': 'ridge_voice_log_a100', 'kind': 'ridge', 'cols': 'voice', 'alpha': 100.0, 'log': True})
    specs.append({'name': 'hgb_shallow_all', 'kind': 'hgb', 'cols': 'all'})
    specs.append({'name': 'hgb_shallow_voice', 'kind': 'hgb', 'cols': 'voice'})
    return specs


def fit_spec(spec, X, y, names, seed):
    kind = spec['kind']
    if kind == 'const':
        if spec['stat'] == 'median':
            c = float(np.median(y))
        else:
            c = float(np.mean(y))

        def pred_const(Xe):
            return np.full(Xe.shape[0], c)
        return pred_const
    cols = col_subset(names, spec['cols'])
    sub_names = [names[i] for i in cols]
    lidx = log_index(sub_names)
    if kind == 'ridge':
        Z = X[:, cols]
        if spec['log']:
            Z = apply_log(Z, lidx)
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        model = Ridge(alpha=spec['alpha'])
        model.fit((Z - mu) / sd, y)

        def pred_ridge(Xe):
            Ze = Xe[:, cols]
            if spec['log']:
                Ze = apply_log(Ze, lidx)
            return model.predict((Ze - mu) / sd)
        return pred_ridge
    if kind == 'hgb':
        model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_depth=3, max_leaf_nodes=8, min_samples_leaf=150, l2_regularization=10.0, random_state=seed)
        model.fit(X[:, cols], y)

        def pred_hgb(Xe):
            return model.predict(Xe[:, cols])
        return pred_hgb
    raise ValueError('unknown spec kind')


def group_mae(y, p, g):
    vals = []
    for gid in np.unique(g):
        m = g == gid
        vals.append(float(np.mean(np.abs(y[m] - p[m]))))
    return float(np.mean(vals))


def evaluate(spec, X, y, g, names, seed):
    n = len(y)
    ng = len(np.unique(g))
    k = min(7, ng)
    oof = np.zeros(n)
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(X, y, g):
        f = fit_spec(spec, X[tr], y[tr], names, seed)
        oof[te] = f(X[te])
    grmse = float(np.sqrt(np.mean((y - oof) ** 2)))
    gmae = group_mae(y, oof, g)
    oof_r = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        f = fit_spec(spec, X[tr], y[tr], names, seed)
        oof_r[te] = f(X[te])
    rrmse = float(np.sqrt(np.mean((y - oof_r) ** 2)))
    return grmse, gmae, rrmse


def between_patient_fraction(y, g):
    tot = float(np.var(y))
    if tot <= 0:
        return 0.0
    means = np.zeros(len(y))
    for gid in np.unique(g):
        m = g == gid
        means[m] = np.mean(y[m])
    return float(np.var(means) / tot)


def fit_predict(train, inputs, seed):
    if seed is None:
        seed = 0
    seed = int(seed)
    names = list(train['feature_names'])
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    g = np.asarray(train['groups'])
    Xe = np.asarray(inputs['X'], dtype=float)
    specs = make_specs()
    table = []
    for spec in specs:
        try:
            grmse, gmae, rrmse = evaluate(spec, X, y, g, names, seed)
            table.append({'name': spec['name'], 'ok': True, 'group_rmse': round(grmse, 4), 'group_mae': round(gmae, 4), 'row_rmse': round(rrmse, 4), 'row_minus_group_gap': round(rrmse - grmse, 4)})
        except Exception as err:
            table.append({'name': spec['name'], 'ok': False, 'error': str(err)[:200]})
    ok = [t for t in table if t['ok'] and np.isfinite(t['group_rmse'])]
    fallback = False
    chosen = None
    pred = None
    refit_error = None
    if len(ok) > 0:
        best = min(ok, key=lambda t: t['group_rmse'])
        chosen = best['name']
        try:
            spec = [s for s in specs if s['name'] == chosen][0]
            f = fit_spec(spec, X, y, names, seed)
            pred = np.asarray(f(Xe), dtype=float)
        except Exception as err:
            refit_error = str(err)[:200]
            pred = None
    if pred is None or pred.shape[0] != Xe.shape[0]:
        fallback = True
        pred = np.full(Xe.shape[0], float(np.median(y)))
    bad = ~np.isfinite(pred)
    n_bad = int(bad.sum())
    if n_bad > 0:
        pred[bad] = float(np.median(y))
    med_row = [t for t in ok if t['name'] == 'median']
    med_group_rmse = med_row[0]['group_rmse'] if len(med_row) > 0 else None
    findings = {
        'selection_rule': 'lowest 7-fold GroupKFold RMSE over train patients, ties earliest in candidate order',
        'chosen': chosen,
        'fallback_used_train_median': fallback,
        'refit_error': refit_error,
        'n_nonfinite_replaced': n_bad,
        'n_train_rows': int(X.shape[0]),
        'n_train_patients': int(len(np.unique(g))),
        'n_eval_rows': int(Xe.shape[0]),
        'train_y_median': round(float(np.median(y)), 4),
        'train_y_mean': round(float(np.mean(y)), 4),
        'train_y_std': round(float(np.std(y)), 4),
        'between_patient_variance_fraction_train': round(between_patient_fraction(y, g), 4),
        'median_group_rmse': med_group_rmse,
        'candidates': table,
        'transductive_use': 'none; all scaling and selection statistics computed from train rows only',
        'note': 'row-level CV is patient-leaky and shown only as a contrast to patient-held-out CV; no evaluation labels used'
    }
    return {'prediction': [float(v) for v in pred], 'findings': findings}
