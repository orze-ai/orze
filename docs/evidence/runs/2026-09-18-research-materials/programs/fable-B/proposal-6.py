import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

SEED = 1729
M = 10.0
K = 5


def make_hgb():
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=SEED)


def elem_stats(C, y):
    pres = C > 0
    pf = pres.astype(float)
    n = pf.sum(axis=0)
    ly = np.log1p(np.maximum(y, 0.0))
    gy = float(y.mean())
    gly = float(ly.mean())
    gq = float(np.percentile(y, 75))
    sy = pf.T @ y
    sly = pf.T @ ly
    enc_y = (sy + M * gy) / (n + M)
    enc_ly = (sly + M * gly) / (n + M)
    enc_q = np.full(C.shape[1], gq)
    for j in range(C.shape[1]):
        if n[j] >= 3:
            q = float(np.percentile(y[pres[:, j]], 75))
            enc_q[j] = (n[j] * q + M * gq) / (n[j] + M)
    return {'n': n, 'y': enc_y, 'ly': enc_ly, 'q': enc_q}


def row_feats(C, st):
    pres = C > 0
    cnt = np.maximum(pres.sum(axis=1), 1).astype(float)
    out = []
    for key in ('y', 'ly', 'q'):
        e = st[key]
        wmean = C @ e
        mx = np.max(np.where(pres, e[None, :], -np.inf), axis=1)
        mn = np.min(np.where(pres, e[None, :], np.inf), axis=1)
        umean = np.where(pres, e[None, :], 0.0).sum(axis=1) / cnt
        out += [wmean, mx, mn, umean]
    logn = np.log1p(st['n'])
    out.append(C @ logn)
    out.append(np.min(np.where(pres, logn[None, :], np.inf), axis=1))
    F = np.column_stack(out)
    F[~np.isfinite(F)] = 0.0
    return F


def te_oof(C, y, groups, k=K):
    F = None
    gkf = GroupKFold(n_splits=k)
    for tr, va in gkf.split(C, y, groups):
        st = elem_stats(C[tr], y[tr])
        f = row_feats(C[va], st)
        if F is None:
            F = np.zeros((C.shape[0], f.shape[1]))
        F[va] = f
    return F


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def metrics(y, p):
    if len(y) == 0:
        return {'n': 0}
    r = y - p
    return {'n': int(len(y)), 'rmse': rmse(y, p), 'mae': float(np.mean(np.abs(r))),
            'bias_y_minus_pred': float(np.mean(r))}


def subsets(y, p, cup):
    return {'all': metrics(y, p), 'cuprate': metrics(y[cup], p[cup]), 'noncuprate': metrics(y[~cup], p[~cup]),
            'y_gt40': metrics(y[y > 40], p[y > 40]), 'y_lt10': metrics(y[y < 10], p[y < 10]),
            'pred_gt40': metrics(y[p > 40], p[p > 40])}


def fit_predict(train, inputs, seed):
    X = np.asarray(train['X'], dtype=float)
    C = np.asarray(train['C'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    groups = np.asarray(train['groups'])
    Xe = np.asarray(inputs['X'], dtype=float)
    Ce = np.asarray(inputs['C'], dtype=float)
    elems = list(train['elements'])
    icu = elems.index('Cu') if 'Cu' in elems else -1
    io = elems.index('O') if 'O' in elems else -1
    if icu >= 0 and io >= 0:
        cup = (C[:, icu] > 0) & (C[:, io] > 0)
    else:
        cup = np.zeros(len(y), dtype=bool)
    base = np.hstack([X, C])
    oof_b = np.zeros(len(y))
    oof_t = np.zeros(len(y))
    fold_rmse = []
    gkf = GroupKFold(n_splits=K)
    for tr, va in gkf.split(base, y, groups):
        mb = make_hgb().fit(base[tr], y[tr])
        pb = np.maximum(mb.predict(base[va]), 0.0)
        oof_b[va] = pb
        Ftr = te_oof(C[tr], y[tr], groups[tr])
        st = elem_stats(C[tr], y[tr])
        Fva = row_feats(C[va], st)
        mt = make_hgb().fit(np.hstack([base[tr], Ftr]), y[tr])
        pt = np.maximum(mt.predict(np.hstack([base[va], Fva])), 0.0)
        oof_t[va] = pt
        fold_rmse.append([rmse(y[va], pb), rmse(y[va], pt)])
    Ftr = te_oof(C, y, groups)
    st = elem_stats(C, y)
    Fe = row_feats(Ce, st)
    model = make_hgb().fit(np.hstack([base, Ftr]), y)
    pred = np.maximum(model.predict(np.hstack([Xe, Ce, Fe])), 0.0)
    fr = np.array(fold_rmse)
    findings = {
        'deployed': 'HGB(250,0.08,31 leaves,l2=1,seed 1729) on X+C+14 element-target-encoding features; train rows encoded OOF by element set (GroupKFold 5), eval rows encoded with full-train element statistics; predictions clipped at 0',
        'note': 'All CV numbers are train-only GroupKFold by element set (nested encoding inside each outer fold), not development scores. Seed argument ignored. Deployed variant pre-declared regardless of CV outcome.',
        'n_train_rows': int(len(y)), 'n_eval_rows': int(len(pred)), 'n_te_features': int(Fe.shape[1]),
        'cv_base_XC': subsets(y, oof_b, cup),
        'cv_XC_plus_elemTE': subsets(y, oof_t, cup),
        'per_fold_rmse_base_vs_te': [[float(a), float(b)] for a, b in fold_rmse],
        'per_fold_rmse_delta_te_minus_base': [float(d) for d in (fr[:, 1] - fr[:, 0])],
        'corr_oof_wmean_y_vs_y_train': float(np.corrcoef(Ftr[:, 0], y)[0, 1]),
        'train_oof_wmean_y_mean': float(Ftr[:, 0].mean()), 'eval_wmean_y_mean': float(Fe[:, 0].mean()),
        'train_frac_rows_with_element_unseen_in_encoding_folds': float(np.mean(Ftr[:, -1] == 0.0)),
        'eval_frac_rows_with_element_unseen_in_train': float(np.mean(Fe[:, -1] == 0.0)),
        'eval_pred_mean': float(pred.mean()), 'eval_pred_frac_gt40': float(np.mean(pred > 40)),
    }
    return {'prediction': pred.tolist(), 'findings': findings}
