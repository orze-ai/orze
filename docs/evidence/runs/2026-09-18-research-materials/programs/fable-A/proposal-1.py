import numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

def _hgb(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)

def _r(x, k=3):
    try:
        x = float(x)
        if not np.isfinite(x):
            return None
        return round(x, k)
    except Exception:
        return None

def _rmse(y, p):
    return _r(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))

def _mae(y, p):
    return _r(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))

def _gmae(y, p, g):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ug, inv = np.unique(np.asarray(g), return_inverse=True)
    s = np.zeros(len(ug)); c = np.zeros(len(ug))
    np.add.at(s, inv, np.abs(y - p)); np.add.at(c, inv, 1.0)
    return _r(np.mean(s / c))

def _scores(y, p, g):
    return {'rmse': _rmse(y, p), 'mae': _mae(y, p), 'group_mae': _gmae(y, p, g)}

def _find_base(history, n):
    try:
        for h in history:
            if not isinstance(h, dict):
                continue
            if h.get('task_id') != 'idea-baseline-2':
                continue
            p = h.get('prediction')
            if isinstance(p, dict):
                p = p.get('prediction', p.get('values'))
            if isinstance(p, (list, tuple)) and len(p) == n:
                return np.asarray(p, float)
    except Exception:
        return None
    return None

def analyze(data, history, seed):
    tr = data['train']; dv = data['development']
    Xtr = np.asarray(tr['X'], float); Ctr = np.asarray(tr['C'], float); ytr = np.asarray(tr['y'], float); gtr = np.asarray(tr['groups'])
    Xdv = np.asarray(dv['X'], float); Cdv = np.asarray(dv['C'], float); ydv = np.asarray(dv['y'], float); gdv = np.asarray(dv['groups'])
    el = list(tr['elements'])
    out = {}
    def qs(y):
        return {'mean': _r(y.mean(), 2), 'median': _r(np.median(y), 2), 'std': _r(y.std(), 2), 'frac_gt50': _r((y > 50).mean()), 'frac_le5': _r((y <= 5).mean())}
    out['sizes'] = {'train_rows': int(len(ytr)), 'train_groups': int(len(set(gtr.tolist()))), 'dev_rows': int(len(ydv)), 'dev_groups': int(len(set(gdv.tolist())))}
    out['y_train'] = qs(ytr); out['y_dev'] = qs(ydv)
    d = defaultdict(list)
    for i, r in enumerate(Ctr):
        d[tuple(np.round(r, 6).tolist())].append(i)
    loo = []
    ndup = 0
    for v in d.values():
        if len(v) > 1:
            ndup += len(v); yy = ytr[v]; s = yy.sum(); n = len(v)
            for i in range(n):
                loo.append((s - yy[i]) / (n - 1) - yy[i])
    loo = np.asarray(loo, float)
    out['train_repeated_compositions'] = {'unique_compositions': int(len(d)), 'rows_in_repeats': int(ndup), 'frac_rows_in_repeats': _r(ndup / len(ytr)), 'loo_same_comp_rmse': _rmse(loo, 0.0) if len(loo) else None, 'loo_same_comp_mae': _mae(loo, 0.0) if len(loo) else None, 'frac_repeat_absdiff_gt10K': _r((np.abs(loo) > 10).mean()) if len(loo) else None}
    def idx(n):
        return el.index(n) if n in el else None
    iCu, iO, iFe = idx('Cu'), idx('O'), idx('Fe')
    def fam(C):
        f = np.full(len(C), 'other', dtype=object)
        if iFe is not None:
            f[C[:, iFe] > 0] = 'Fe'
        if iCu is not None and iO is not None:
            f[(C[:, iCu] > 0) & (C[:, iO] > 0)] = 'CuO'
        return f
    ftr = fam(Ctr); fdv = fam(Cdv)
    out['families'] = {}
    for k in ['CuO', 'Fe', 'other']:
        mt = ftr == k; md = fdv == k
        out['families'][k] = {'train_frac': _r(mt.mean()), 'dev_frac': _r(md.mean()), 'train_meanTc': _r(ytr[mt].mean(), 2) if mt.any() else None, 'dev_meanTc': _r(ydv[md].mean(), 2) if md.any() else None}
    trcount = (Ctr > 0).sum(0)
    unseen = trcount == 0; rare = trcount < 20
    dv_unseen = ((Cdv > 0) & unseen[None, :]).any(1); dv_rare = ((Cdv > 0) & rare[None, :]).any(1)
    out['element_coverage'] = {'elements_absent_from_train': [el[i] for i in np.where(unseen)[0]][:25], 'dev_rows_with_absent_element': int(dv_unseen.sum()), 'dev_rows_with_rare_element_lt20_train_rows': int(dv_rare.sum())}
    Ctn = Ctr / (np.linalg.norm(Ctr, axis=1, keepdims=True) + 1e-12)
    Cdn = Cdv / (np.linalg.norm(Cdv, axis=1, keepdims=True) + 1e-12)
    maxsim = np.zeros(len(Cdv)); knn = np.zeros(len(Cdv))
    for s in range(0, len(Cdv), 500):
        S = Cdn[s:s + 500] @ Ctn.T
        top = np.argpartition(-S, 5, axis=1)[:, :5]
        knn[s:s + 500] = ytr[top].mean(1)
        maxsim[s:s + 500] = S.max(1)
    out['knn5_cosine_on_C_dev'] = _scores(ydv, knn, gdv)
    base = _find_base(history, len(ydv))
    base_src = 'history_idea-baseline-2'
    if base is None:
        m = _hgb(seed); m.fit(Xtr, ytr); base = m.predict(Xdv); base_src = 'refit_HGB_X_not_official'
    out['hgb_X_dev_source'] = base_src
    out['hgb_X_dev'] = _scores(ydv, base, gdv)
    res = base - ydv
    out['hgb_X_dev_by_family'] = {k: {'n': int((fdv == k).sum()), 'rmse': _rmse(ydv[fdv == k], base[fdv == k]), 'bias': _r(res[fdv == k].mean(), 2)} for k in ['CuO', 'Fe', 'other'] if (fdv == k).any()}
    bins = [(-1, 10), (10, 40), (40, 80), (80, 1e9)]
    out['hgb_X_dev_by_true_Tc'] = {}
    for lo, hi in bins:
        m_ = (ydv > lo) & (ydv <= hi)
        if m_.any():
            out['hgb_X_dev_by_true_Tc']['%g-%g' % (lo, hi)] = {'n': int(m_.sum()), 'rmse': _rmse(ydv[m_], base[m_]), 'bias': _r(res[m_].mean(), 2)}
    qcut = np.quantile(maxsim, [0.25, 0.5, 0.75])
    edges = [-1.0] + qcut.tolist() + [2.0]
    out['hgb_X_dev_by_max_cosine_sim_to_train_quartile'] = {}
    for i in range(4):
        m_ = (maxsim > edges[i]) & (maxsim <= edges[i + 1])
        if m_.any():
            out['hgb_X_dev_by_max_cosine_sim_to_train_quartile']['q%d_sim_le_%.3f' % (i + 1, edges[i + 1])] = {'n': int(m_.sum()), 'rmse_hgb': _rmse(ydv[m_], base[m_]), 'rmse_knn5': _rmse(ydv[m_], knn[m_])}
    a = np.abs(res); k5 = max(1, int(0.05 * len(a))); topi = np.argsort(-a)[:k5]
    out['hgb_X_dev_error_concentration'] = {'top5pct_rows_share_of_sq_error': _r((res[topi] ** 2).sum() / (res ** 2).sum()), 'rmse_excluding_top5pct': _rmse(np.delete(ydv, topi), np.delete(base, topi))}
    out['hgb_X_dev_by_element_coverage'] = {'rows_with_absent_element_rmse': _rmse(ydv[dv_unseen], base[dv_unseen]) if dv_unseen.any() else None, 'rows_with_rare_element_rmse': _rmse(ydv[dv_rare], base[dv_rare]) if dv_rare.any() else None, 'rows_all_common_rmse': _rmse(ydv[~dv_rare], base[~dv_rare]) if (~dv_rare).any() else None}
    reps = {'X': (Xtr, Xdv), 'C': (Ctr, Cdv), 'X+C': (np.hstack([Xtr, Ctr]), np.hstack([Xdv, Cdv]))}
    gkf = GroupKFold(n_splits=5)
    out['representation_contrast'] = {}
    for name, (A, B) in reps.items():
        for tgt in ['raw', 'log1p']:
            if tgt == 'log1p' and name != 'X+C':
                continue
            cvp = np.zeros(len(ytr))
            for tri, tei in gkf.split(A, ytr, gtr):
                m = _hgb(seed)
                if tgt == 'raw':
                    m.fit(A[tri], ytr[tri]); cvp[tei] = m.predict(A[tei])
                else:
                    m.fit(A[tri], np.log1p(ytr[tri])); cvp[tei] = np.expm1(m.predict(A[tei]))
            m = _hgb(seed)
            if tgt == 'raw':
                m.fit(A, ytr); dp = m.predict(B)
            else:
                m.fit(A, np.log1p(ytr)); dp = np.expm1(m.predict(B))
            out['representation_contrast'][name + '_' + tgt] = {'train_groupcv5': _scores(ytr, cvp, gtr), 'dev': _scores(ydv, dp, gdv), 'dev_rmse_CuO': _rmse(ydv[fdv == 'CuO'], dp[fdv == 'CuO']) if (fdv == 'CuO').any() else None, 'dev_rmse_other': _rmse(ydv[fdv == 'other'], dp[fdv == 'other']) if (fdv == 'other').any() else None}
    out['interpretation_note'] = 'Calculated quantities above; families are heuristic C-based labels (Cu and O present -> CuO, Fe present -> Fe, else other), not chemical ground truth. Repeated-composition LOO error is a noise-floor estimate only for repeated rows. Dev-fit numbers use development labels for analysis only and are not official method scores.'
    return out
