import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

def _hgb(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)

def _blocks(X, C):
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    P = (C > 0).astype(float)
    return {'X': X, 'C': C, 'XC': np.hstack([X, C]), 'XP': np.hstack([X, P])}

def fit_predict(train, inputs, seed):
    seed = int(seed) if seed is not None else 0
    y = np.asarray(train['y'], dtype=float)
    groups = np.asarray(train['groups'])
    tb = _blocks(train['X'], train['C'])
    eb = _blocks(inputs['X'], inputs['C'])
    n = len(y)
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(tb['X'], y, groups))
    cv = {}
    oof = {}
    for name in ['X', 'C', 'XC', 'XP']:
        F = tb[name]
        p_all = np.zeros(n)
        fold_rmse = []
        for tr, te in folds:
            m = _hgb(seed).fit(F[tr], y[tr])
            p = m.predict(F[te])
            p_all[te] = p
            fold_rmse.append(float(np.sqrt(np.mean((p - y[te]) ** 2))))
        oof[name] = p_all
        pc = np.clip(p_all, 0.0, None)
        cv[name] = {'rmse': float(np.sqrt(np.mean((p_all - y) ** 2))), 'mae': float(np.mean(np.abs(p_all - y))), 'rmse_clip0': float(np.sqrt(np.mean((pc - y) ** 2))), 'fold_rmse': fold_rmse}
    bins = [(0, 10), (10, 40), (40, 200)]
    by_bin = {}
    for lo, hi in bins:
        mk = (y >= lo) & (y < hi)
        if mk.sum() > 0:
            by_bin['%d-%d' % (lo, hi)] = {'n': int(mk.sum()), 'rmse_X': float(np.sqrt(np.mean((oof['X'][mk] - y[mk]) ** 2))), 'rmse_XC': float(np.sqrt(np.mean((oof['XC'][mk] - y[mk]) ** 2))), 'mean_bias_XC': float(np.mean(oof['XC'][mk] - y[mk]))}
    Ct = np.round(tb['C'], 6)
    keys = {}
    for i in range(n):
        keys.setdefault(Ct[i].tobytes(), []).append(i)
    dup = [v for v in keys.values() if len(v) > 1]
    n_dup_rows = int(sum(len(v) for v in dup))
    if n_dup_rows > 0:
        res = np.concatenate([y[v] - np.mean(y[v]) for v in dup])
        floor_all = float(np.sqrt(np.sum(res ** 2) / n))
        floor_dup = float(np.sqrt(np.mean(res ** 2)))
        med_spread = float(np.median([np.max(y[v]) - np.min(y[v]) for v in dup]))
    else:
        floor_all = 0.0
        floor_dup = 0.0
        med_spread = 0.0
    train_present = (tb['C'] > 0).any(axis=0)
    eval_present = (eb['C'] > 0).any(axis=0)
    unseen_el = [inputs['elements'][j] for j in range(len(eval_present)) if eval_present[j] and not train_present[j]]
    rows_with_unseen = int(((eb['C'] > 0) & (~train_present)[None, :]).any(axis=1).sum())
    final_name = 'XC'
    m = _hgb(seed).fit(tb[final_name], y)
    pred = m.predict(eb[final_name])
    findings = {'design': 'HGB fixed (250 it, lr 0.08, 31 leaves, l2 1); only representation varies; 5-fold GroupKFold by element set on train', 'train_rows': int(n), 'train_groups': int(len(set(groups.tolist()))), 'cv_by_representation': cv, 'oof_rmse_by_true_tc_bin': by_bin, 'duplicate_composition_rows_train': n_dup_rows, 'duplicate_groups_train': int(len(dup)), 'noise_floor_rmse_dup_rows_only': floor_dup, 'noise_floor_rmse_spread_over_all_rows': floor_all, 'median_tc_range_within_duplicates': med_spread, 'eval_elements_absent_from_train': unseen_el, 'eval_rows_with_element_absent_from_train': rows_with_unseen, 'submitted_representation': final_name, 'submitted_postprocessing': 'none (no clipping)', 'note': 'CV numbers are train-only group-held-out estimates, not development scores; representation CV order is a calculated finding, causes are interpretation'}
    return {'prediction': [float(v) for v in pred], 'findings': findings}
