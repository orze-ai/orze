import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


def design(X, cols, log_cols, floors):
    Z = X[:, cols].astype(float).copy()
    for j, c in enumerate(cols):
        if c in log_cols:
            Z[:, j] = np.log(np.maximum(Z[:, j], floors[c]))
    return Z


def fit_pred(Ztr, ytr, w, Zte, alpha):
    mu = Ztr.mean(0)
    sd = Ztr.std(0)
    sd[sd == 0] = 1.0
    m = Ridge(alpha=alpha).fit((Ztr - mu) / sd, ytr, sample_weight=w)
    return m.predict((Zte - mu) / sd)


def weights(g, mode):
    if mode == 'row':
        return np.ones(len(g))
    ug, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w * len(g) / w.sum()


def core(X, y, g, names, Xe, med):
    n_e = len(Xe)
    d = [i for i, n in enumerate(names) if n in ('age', 'sex')]
    t = [i for i, n in enumerate(names) if 'test_time' in n]
    v = [i for i in range(len(names)) if i not in d and i not in t]
    logc = set(i for i in v if any(k in names[i] for k in ('jitter', 'shimmer', 'nhr')) and np.all(X[:, i] > 0))
    floors = {i: 0.5 * float(X[:, i].min()) for i in logc}
    sets = {'demo': d, 'demo_time': d + t, 'voice_raw': v, 'voice_log': v, 'all_raw': d + t + v, 'all_log': d + t + v}
    nolog = set(['voice_raw', 'all_raw'])
    alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    folds = list(GroupKFold(n_splits=7).split(X, y, g))
    ug, inv = np.unique(g, return_inverse=True)
    K = len(ug)
    pm_y = np.array([y[inv == k].mean() for k in range(K)])
    pm_X = np.array([X[inv == k].mean(0) for k in range(K)])
    corr = {}
    for i, n in enumerate(names):
        col = pm_X[:, i]
        corr[n] = round(float(np.corrcoef(col, pm_y)[0, 1]), 3) if col.std() > 0 else None
    ymin, ymax = float(y.min()), float(y.max())

    def cv_eval(pred):
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        gm = float(np.mean([np.mean(np.abs(pred[inv == k] - y[inv == k])) for k in range(K)]))
        return rmse, gm

    results = []
    pred = np.zeros(len(y))
    for tr, te in folds:
        pred[te] = np.median(y[tr])
    r, gm = cv_eval(pred)
    med_res = {'model': 'median', 'set': '-', 'alpha': None, 'w': '-', 'cv_rmse': round(r, 4), 'cv_gmae': round(gm, 4)}
    results.append(med_res)
    for sname, cols in sets.items():
        if not cols:
            continue
        lc = set() if sname in nolog else logc
        Z = design(X, cols, lc, floors)
        for wm in ('row', 'patient'):
            w = weights(g, wm)
            for a in alphas:
                pred = np.zeros(len(y))
                for tr, te in folds:
                    pred[te] = fit_pred(Z[tr], y[tr], w[tr], Z[te], a)
                pred = np.clip(pred, ymin, ymax)
                r, gm = cv_eval(pred)
                results.append({'model': 'ridge', 'set': sname, 'alpha': a, 'w': wm, 'cv_rmse': round(r, 4), 'cv_gmae': round(gm, 4)})
    results.sort(key=lambda z: z['cv_rmse'])
    best = results[0]
    best_per = {}
    for z in results:
        key = z['set'] + '/' + z['w']
        if key not in best_per:
            best_per[key] = z
    if best['model'] == 'median':
        out = np.full(n_e, med)
    else:
        cols = sets[best['set']]
        lc = set() if best['set'] in nolog else logc
        Z = design(X, cols, lc, floors)
        Ze = design(Xe, cols, lc, floors)
        out = np.clip(fit_pred(Z, y, weights(g, best['w']), Ze, best['alpha']), ymin, ymax)
    findings = {
        'fallback_used': False,
        'selection_rule': 'min RMSE in 7-fold GroupKFold over 28 train patients; refit on all train',
        'chosen': best,
        'median_cv': med_res,
        'top12': results[:12],
        'best_per_set_weighting': best_per,
        'patient_level_corr_mean_feature_vs_mean_y_n28': corr,
        'n_train_patients': int(K),
        'log_transformed': [names[i] for i in sorted(logc)],
        'transductive_use_of_inputs': 'none; scaling, log floors and clipping range from train only',
        'pred_summary': {'mean': round(float(np.mean(out)), 3), 'std': round(float(np.std(out)), 3), 'min': round(float(np.min(out)), 3), 'max': round(float(np.max(out)), 3), 'train_median': round(med, 3)},
    }
    return {'prediction': [float(x) for x in out], 'findings': findings}


def fit_predict(train, inputs, seed):
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    g = np.asarray([str(v) for v in train['groups']])
    names = [str(n).lower() for n in train['feature_names']]
    Xe = np.asarray(inputs['X'], dtype=float)
    med = float(np.median(y))
    try:
        return core(X, y, g, names, Xe, med)
    except Exception as e:
        return {'prediction': [med] * len(Xe), 'findings': {'fallback_used': True, 'fallback': 'train_median_only', 'error': str(e)[:500]}}
