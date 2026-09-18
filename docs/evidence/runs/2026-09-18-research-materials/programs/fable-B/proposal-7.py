import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

def _hgb(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=int(seed))

def _bag(X, y, groups, Xte, n_bags, seed0):
    rng = np.random.RandomState(seed0)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_group = [[] for _ in range(len(ug))]
    for i, j in enumerate(inv):
        idx_by_group[j].append(i)
    idx_by_group = [np.asarray(v, dtype=int) for v in idx_by_group]
    acc = np.zeros(Xte.shape[0])
    for b in range(n_bags):
        samp = rng.randint(0, len(ug), size=len(ug))
        rows = np.concatenate([idx_by_group[j] for j in samp])
        m = _hgb(seed0 + 1 + b)
        m.fit(X[rows], y[rows])
        acc += m.predict(Xte)
    return acc / n_bags

def _stats(y, p):
    if len(y) == 0:
        return {'n': 0, 'rmse': None, 'mae': None, 'bias_y_minus_pred': None}
    r = y - p
    return {'n': int(len(y)), 'rmse': float(np.sqrt(np.mean(r ** 2))), 'mae': float(np.mean(np.abs(r))), 'bias_y_minus_pred': float(np.mean(r))}

def fit_predict(train, inputs, seed):
    Xraw = np.asarray(train['X'], dtype=float)
    nX = Xraw.shape[1]
    X = np.hstack([Xraw, np.asarray(train['C'], dtype=float)])
    y = np.asarray(train['y'], dtype=float)
    groups = np.asarray(train['groups'])
    Xe = np.hstack([np.asarray(inputs['X'], dtype=float), np.asarray(inputs['C'], dtype=float)])
    elements = list(train['elements'])
    iCu = elements.index('Cu') if 'Cu' in elements else None
    iO = elements.index('O') if 'O' in elements else None
    def cup_mask(M):
        if iCu is None or iO is None:
            return np.zeros(M.shape[0], dtype=bool)
        return (M[:, nX + iCu] > 0) & (M[:, nX + iO] > 0)
    NB_CV = 6
    NB_DEPLOY = 10
    SEED = 1729
    gkf = GroupKFold(n_splits=5)
    oof_single = np.zeros(len(y))
    oof_bag = np.zeros(len(y))
    fold_rows = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = _hgb(SEED)
        m.fit(X[tr], y[tr])
        ps = np.clip(m.predict(X[te]), 0, None)
        pb = np.clip(_bag(X[tr], y[tr], groups[tr], X[te], NB_CV, SEED + 100 * k), 0, None)
        oof_single[te] = ps
        oof_bag[te] = pb
        fold_rows.append([float(np.sqrt(np.mean((y[te] - ps) ** 2))), float(np.sqrt(np.mean((y[te] - pb) ** 2)))])
    cm = cup_mask(X)
    hi = y > 40
    lo = y < 10
    def block(p):
        return {'all': _stats(y, p), 'cuprate': _stats(y[cm], p[cm]), 'noncuprate': _stats(y[~cm], p[~cm]), 'y_gt40': _stats(y[hi], p[hi]), 'y_lt10': _stats(y[lo], p[lo])}
    pe = np.clip(_bag(X, y, groups, Xe, NB_DEPLOY, SEED), 0, None)
    cme = cup_mask(Xe)
    findings = {
        'deployed': 'mean of %d HGB(250,0.08,31 leaves,l2=1) fit on element-set (group) bootstrap resamples of train X+C; predictions clipped at 0; fixed seed 1729, seed argument ignored; pre-declared regardless of CV outcome' % NB_DEPLOY,
        'cv_single_XC': block(oof_single),
        'cv_groupbag%d_XC' % NB_CV: block(oof_bag),
        'per_fold_rmse_single_vs_bag': fold_rows,
        'per_fold_rmse_delta_bag_minus_single': [r[1] - r[0] for r in fold_rows],
        'oof_pred_corr_single_bag': float(np.corrcoef(oof_single, oof_bag)[0, 1]),
        'oof_pred_rms_diff_single_bag': float(np.sqrt(np.mean((oof_single - oof_bag) ** 2))),
        'n_train_rows': int(len(y)),
        'n_train_groups': int(len(np.unique(groups))),
        'n_eval_rows': int(Xe.shape[0]),
        'eval_frac_cuprate_by_C': float(np.mean(cme)),
        'eval_pred_mean': float(np.mean(pe)),
        'eval_pred_frac_gt40': float(np.mean(pe > 40)),
        'note': 'CV numbers are train-only GroupKFold(5) by element set, not development scores. Group bootstrap resamples element sets with replacement, so each member omits about 37 percent of train systems. Only the fitting procedure differs from the single-HGB arm; features, hyperparameters and clipping are identical.'
    }
    return {'prediction': pe.tolist(), 'findings': findings}
