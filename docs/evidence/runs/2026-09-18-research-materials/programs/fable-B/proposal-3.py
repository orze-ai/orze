import math
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

def _r(x, d=2):
    try:
        x = float(x)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, d)

def _rmse(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    if len(p) == 0:
        return None
    return _r(math.sqrt(float(np.mean((p - y) ** 2))))

def _records(h):
    if isinstance(h, list):
        return h
    if isinstance(h, dict):
        for k in ('actions', 'records', 'history', 'tasks'):
            v = h.get(k)
            if isinstance(v, list):
                return v
        return [h]
    return []

def _pred(rec, n):
    if not isinstance(rec, dict):
        return None
    p = rec.get('prediction')
    if isinstance(p, dict):
        p = p.get('prediction')
    if p is None:
        return None
    try:
        arr = np.asarray(p, float).ravel()
    except Exception:
        return None
    if arr.shape[0] != n or not np.all(np.isfinite(arr)):
        return None
    return arr

def analyze(data, history, seed):
    tr = data['train']; dv = data['development']
    Ctr = np.asarray(tr['C'], float); Cdv = np.asarray(dv['C'], float)
    ytr = np.asarray(tr['y'], float); ydv = np.asarray(dv['y'], float)
    elems = list(tr['elements'])
    n = len(ydv)
    gtr = list(tr['groups']); gdv = list(dv['groups'])
    def gidx(groups):
        d = {}
        for i, g in enumerate(groups):
            d.setdefault(g, []).append(i)
        return d
    gi_tr = gidx(gtr); gi_dv = gidx(gdv)
    tr_names = list(gi_tr.keys()); dv_names = list(gi_dv.keys())
    Str = np.stack([(Ctr[gi_tr[g][0]] > 0).astype(float) for g in tr_names])
    Sdv = np.stack([(Cdv[gi_dv[g][0]] > 0).astype(float) for g in dv_names])
    tr_med = np.array([np.median(ytr[gi_tr[g]]) for g in tr_names])
    dv_setid = np.zeros(n, int)
    for si, g in enumerate(dv_names):
        dv_setid[gi_dv[g]] = si
    inter = Sdv @ Str.T
    sz_dv = Sdv.sum(1); sz_tr = Str.sum(1)
    jac = inter / (sz_dv[:, None] + sz_tr[None, :] - inter)
    maxjac_set = jac.max(1)
    sup_in_tr = ((inter == sz_dv[:, None]) & (sz_tr[None, :] > sz_dv[:, None])).any(1)
    sub_in_tr = ((inter == sz_tr[None, :]) & (sz_tr[None, :] < sz_dv[:, None])).any(1)
    maxjac = maxjac_set[dv_setid]
    rel = (sup_in_tr | sub_in_tr)[dv_setid]
    edges = [0.34, 0.5, 0.67, 0.8]
    bin_id = np.digitize(maxjac, edges)
    bl = ['<0.34', '0.34-0.5', '0.5-0.67', '0.67-0.8', '>=0.8']
    def has(e):
        return (Cdv[:, elems.index(e)] > 0) if e in elems else np.zeros(n, bool)
    cu = has('Cu') & has('O')
    fe = has('Fe') & (has('As') | has('Se') | has('Te') | has('P') | has('S')) & ~cu
    mgb = has('Mg') & has('B') & ~cu & ~fe
    oxide = has('O') & ~cu & ~fe & ~mgb
    rest = ~(cu | fe | mgb | oxide)
    fams = {'cuprate': cu, 'Fe_pn_ch': fe, 'MgB': mgb, 'oth_oxide': oxide, 'rest': rest}
    preds = {}
    preds['train_median'] = np.full(n, float(np.median(ytr)))
    order = np.argsort(-jac, axis=1)
    for k in (1, 5, 20):
        kk = min(k, jac.shape[1])
        idx = order[:, :kk]
        w = np.take_along_axis(jac, idx, axis=1) + 1e-9
        ps = (tr_med[idx] * w).sum(1) / w.sum(1)
        preds['jac_knn_k%d' % k] = ps[dv_setid]
    nn = NearestNeighbors(n_neighbors=5, metric='manhattan', algorithm='brute').fit(Ctr)
    dist, nidx = nn.kneighbors(Cdv)
    preds['compL1_1nn'] = ytr[nidx[:, 0]]
    preds['compL1_5nn'] = ytr[nidx].mean(1)
    models = {}
    for rec in _records(history):
        p = _pred(rec, n)
        if p is None or not rec.get('valid', True):
            continue
        name = str(rec.get('task_id') or rec.get('action_id') or ('hist%d' % len(models)))[:24]
        if len(models) < 4 and name not in models:
            models[name] = p
    allp = {}
    allp.update(preds); allp.update(models)
    dv_set_med = np.zeros(n)
    for g, ii in gi_dv.items():
        dv_set_med[ii] = np.median(ydv[ii])
    Cr = np.round(Cdv, 6)
    _, inv, cnt = np.unique(Cr, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    dup_mask = cnt[inv] > 1
    dup_ss = 0.0; dup_n = 0
    for u in np.nonzero(cnt > 1)[0]:
        ii = np.nonzero(inv == u)[0]
        dup_ss += float(((ydv[ii] - ydv[ii].mean()) ** 2).sum()); dup_n += len(ii)
    out = {'n_dev_rows': n, 'n_dev_sets': len(dv_names), 'n_train_sets': len(tr_names),
           'dev_rows_with_subset_or_superset_set_in_train': int(rel.sum()),
           'dev_rows_per_overlap_bin': {bl[b]: int((bin_id == b).sum()) for b in range(5)},
           'dev_rows_per_family': {f: int(m.sum()) for f, m in fams.items()},
           'dev_ystd_per_family': {f: (_r(ydv[m].std()) if m.sum() > 1 else None) for f, m in fams.items()},
           'leaky_oracle_dev_set_median_rmse': _rmse(dv_set_med, ydv),
           'dev_exact_dup_composition_rows': int(dup_mask.sum()),
           'dev_dup_composition_y_rms_spread': (_r(math.sqrt(dup_ss / dup_n)) if dup_n else None),
           'history_models_used': list(models.keys())}
    per = {}
    for name, p in allp.items():
        full = name in models
        d = {'rmse': _rmse(p, ydv)}
        d['by_overlap'] = {bl[b]: _rmse(p[bin_id == b], ydv[bin_id == b]) for b in range(5) if (bin_id == b).sum() > 0}
        d['by_family'] = {f: _rmse(p[m], ydv[m]) for f, m in fams.items() if m.sum() > 0}
        pm = np.zeros(n); ym = np.zeros(n)
        for g, ii in gi_dv.items():
            pm[ii] = p[ii].mean(); ym[ii] = ydv[ii].mean()
        d['between_set_rmse'] = _rmse(pm, ym)
        d['within_set_rmse'] = _rmse(p - pm, ydv - ym)
        rhos = []
        for g, ii in gi_dv.items():
            if len(ii) >= 8 and np.std(ydv[ii]) > 0 and np.std(p[ii]) > 0:
                rho = spearmanr(p[ii], ydv[ii])[0]
                if rho == rho:
                    rhos.append(float(rho))
        d['within_spearman_median'] = (_r(np.median(rhos)) if rhos else None)
        if full:
            d['mae'] = _r(np.mean(np.abs(p - ydv)))
            d['n_sets_spearman'] = len(rhos)
            d['within_spearman_frac_gt_0.3'] = (_r(np.mean(np.array(rhos) > 0.3)) if rhos else None)
            d['rel_set_in_train_rmse'] = _rmse(p[rel], ydv[rel]); d['no_rel_set_rmse'] = _rmse(p[~rel], ydv[~rel])
            d['bias_tc_lt10'] = _r(np.mean((p - ydv)[ydv < 10])); d['bias_tc_gt40'] = _r(np.mean((p - ydv)[ydv > 40]))
        per[name] = d
    out['per_predictor'] = per
    if models:
        best = min(models, key=lambda k: float(np.mean((models[k] - ydv) ** 2)))
        p = models[best]
        sse = {}
        for g, ii in gi_dv.items():
            sse[g] = float(((p[ii] - ydv[ii]) ** 2).sum())
        tot = sum(sse.values())
        top = sorted(sse, key=sse.get, reverse=True)[:6]
        out['best_history_model'] = best
        out['top6_sets_share_of_sse'] = (_r(sum(sse[g] for g in top) / tot, 3) if tot > 0 else None)
        out['top6_sets'] = [{'set': g[:30], 'n': len(gi_dv[g]), 'sse_share': _r(sse[g] / tot, 3),
                             'maxjac': _r(maxjac_set[dv_names.index(g)]), 'ymean': _r(ydv[gi_dv[g]].mean()),
                             'pmean': _r(p[gi_dv[g]].mean())} for g in top]
        se = (p - ydv) ** 2
        set_mse = np.array([np.mean(se[gi_dv[g]]) for g in dv_names])
        set_ystd = np.array([ydv[gi_dv[g]].std() for g in dv_names])
        out['spearman_set_mse_vs_maxjac'] = _r(spearmanr(set_mse, maxjac_set)[0], 3)
        out['spearman_set_mse_vs_set_ystd'] = _r(spearmanr(set_mse, set_ystd)[0], 3)
    out['note'] = 'Calculated on actual dev labels. Leaky set-median oracle marks within-set spread only, not an achievable score. Identity controls use no descriptors. Interpretation is separate from these calculations.'
    return out
