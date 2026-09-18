import json, math
import numpy as np
from scipy.stats import spearmanr

def _r(x):
    try:
        return float(round(float(x), 3))
    except Exception:
        return None

def _rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0:
        return None
    return _r(math.sqrt(float(np.mean((a - b) ** 2))))

def analyze(data, history, seed):
    tr = data['train']; dv = data['development']
    Ctr = np.asarray(tr['C'], float); Cdv = np.asarray(dv['C'], float)
    ytr = np.asarray(tr['y'], float); ydv = np.asarray(dv['y'], float)
    gtr = list(tr['groups']); gdv = list(dv['groups']); gdv_arr = np.asarray(gdv)
    els = list(dv['elements'])
    n = len(ydv)
    out = {'design': 'development rows with labels; history predictions of matching length; identity-only controls fit on train only; strata are calculated, causes are interpretation', 'dev_rows': n, 'dev_groups': len(set(gdv))}
    def uniq(groups, P):
        d = {}
        for i, g in enumerate(groups):
            d.setdefault(g, []).append(i)
        keys = list(d.keys())
        Pu = np.asarray([P[d[k][0]] for k in keys], float)
        return d, keys, Pu
    Ptr = (Ctr > 0).astype(float); Pdv = (Cdv > 0).astype(float)
    dtr, ktr, Ptu = uniq(gtr, Ptr)
    ddv, kdv, Pdu = uniq(gdv, Pdv)
    tr_med = np.asarray([np.median(ytr[dtr[k]]) for k in ktr])
    inter = Pdu @ Ptu.T
    szd = Pdu.sum(1); szt = Ptu.sum(1)
    union = szd[:, None] + szt[None, :] - inter
    J = inter / np.maximum(union, 1.0)
    maxJ = J.max(1)
    has_sub = ((inter == szt[None, :]) & (szt[None, :] < szd[:, None])).any(1)
    has_sup = ((inter == szd[:, None]) & (szt[None, :] > szd[:, None])).any(1)
    order = np.argsort(-J, axis=1)
    preds = {}
    row_maxJ = np.zeros(n); row_sub = np.zeros(n, bool); row_sup = np.zeros(n, bool)
    for k in (1, 5, 20):
        idx = order[:, :k]
        w = np.take_along_axis(J, idx, 1) + 1e-6
        pg = (w * tr_med[idx]).sum(1) / w.sum(1)
        pr = np.zeros(n)
        for j, g in enumerate(kdv):
            pr[ddv[g]] = pg[j]
        preds['ctl_jaccardknn%d' % k] = pr
    for j, g in enumerate(kdv):
        ii = ddv[g]; row_maxJ[ii] = maxJ[j]; row_sub[ii] = has_sub[j]; row_sup[ii] = has_sup[j]
    nn1 = np.zeros(n); nn1d = np.zeros(n)
    step = 200
    for s in range(0, n, step):
        blk = Cdv[s:s+step]
        D = np.abs(blk[:, None, :] - Ctr[None, :, :]).sum(2)
        a = D.argmin(1)
        nn1[s:s+step] = ytr[a]; nn1d[s:s+step] = D[np.arange(len(blk)), a]
    preds['ctl_compL1_1nn'] = nn1
    preds['ctl_trainmedian'] = np.full(n, float(np.median(ytr)))
    out['comp_1nn_L1_distance_quartiles'] = [_r(q) for q in np.percentile(nn1d, [25, 50, 75])]
    out['maxJ_quartiles_dev_groups'] = [_r(q) for q in np.percentile(maxJ, [10, 25, 50, 75, 90])]
    out['dev_groups_with_train_subset'] = int(has_sub.sum()); out['dev_groups_with_train_superset'] = int(has_sup.sum())
    hist = []
    for h in (history or []):
        try:
            if not isinstance(h, dict):
                continue
            p = h.get('prediction')
            if p is None:
                p = (h.get('facts') or {}).get('prediction')
            if p is None or len(p) != n:
                continue
            p = np.asarray(p, float)
            if not np.all(np.isfinite(p)):
                continue
            tid = str(h.get('task_id', 'hist'))
            hist.append((tid, p, _rmse(ydv, p)))
        except Exception:
            continue
    hist.sort(key=lambda t: (t[2] if t[2] is not None else 1e9))
    out['history_predictions_found'] = [(t[0], t[2]) for t in hist]
    for tid, p, _ in hist[:3]:
        preds['h_' + tid.replace('idea-', '')[:14]] = p
    def gmae(p):
        v = []
        for g, ii in ddv.items():
            v.append(float(np.mean(np.abs(ydv[ii] - p[ii]))))
        return _r(np.mean(v))
    overall = {}
    big = [g for g, ii in ddv.items() if len(ii) >= 8 and np.std(ydv[ii]) > 1.0]
    out['sets_with_ge8_rows_and_ystd_gt1'] = len(big)
    grp_err_maxJ = None
    for pn, p in preds.items():
        e = ydv - p
        d = {'rmse': _rmse(ydv, p), 'mae': _r(np.mean(np.abs(e))), 'group_mae': gmae(p)}
        d['bias_by_tc'] = {'0-10': _r(np.mean((p - ydv)[ydv < 10])), '10-40': _r(np.mean((p - ydv)[(ydv >= 10) & (ydv < 40)])), '40+': _r(np.mean((p - ydv)[ydv >= 40]))}
        sp = []
        for g in big:
            ii = ddv[g]
            if np.std(p[ii]) < 1e-9:
                sp.append(0.0); continue
            r = spearmanr(p[ii], ydv[ii]).correlation
            sp.append(0.0 if not np.isfinite(r) else float(r))
        if sp:
            sp = np.asarray(sp)
            d['within_set_spearman_median'] = _r(np.median(sp)); d['within_set_spearman_frac_gt0.5'] = _r(np.mean(sp > 0.5)); d['within_set_spearman_frac_lt0'] = _r(np.mean(sp < 0))
        gm = np.asarray([np.mean(np.abs(e[ddv[g]])) for g in kdv])
        d['corr_group_abserr_vs_maxJ'] = _r(np.corrcoef(gm, maxJ)[0, 1])
        overall[pn] = d
    out['overall_by_predictor'] = overall
    def idx_has(sym, C):
        if sym in els:
            return C[:, els.index(sym)] > 0
        return np.zeros(len(C), bool)
    cu = idx_has('Cu', Cdv) & idx_has('O', Cdv)
    fe = idx_has('Fe', Cdv) & (idx_has('As', Cdv) | idx_has('Se', Cdv) | idx_has('Te', Cdv) | idx_has('P', Cdv)) & ~cu
    mgb = idx_has('Mg', Cdv) & idx_has('B', Cdv) & ~cu & ~fe
    ox = idx_has('O', Cdv) & ~cu & ~fe & ~mgb
    rest = ~(cu | fe | mgb | ox)
    fam = {'cuprate_CuO': cu, 'Fe_pnictide_chalc': fe, 'MgB': mgb, 'other_oxide': ox, 'rest_nonoxide': rest}
    jb = {'maxJ<0.5': row_maxJ < 0.5, '0.5-0.67': (row_maxJ >= 0.5) & (row_maxJ < 0.67), '0.67-0.8': (row_maxJ >= 0.67) & (row_maxJ < 0.8), 'maxJ>=0.8': row_maxJ >= 0.8}
    rel = {'has_train_subset': row_sub, 'no_train_subset': ~row_sub, 'has_train_superset': row_sup}
    def table(masks):
        t = {}
        for name, m in masks.items():
            m = np.asarray(m, bool)
            if not m.any():
                t[name] = {'n': 0}; continue
            row = {'n': int(m.sum()), 'g': int(len(set(gdv_arr[m]))), 'ymean': _r(ydv[m].mean()), 'ystd': _r(ydv[m].std())}
            for pn, p in preds.items():
                row[pn] = _rmse(ydv[m], p[m])
            t[name] = row
        return t
    out['rmse_by_maxJ_bin'] = table(jb)
    out['rmse_by_family'] = table(fam)
    out['rmse_by_subset_relation'] = table(rel)
    fam_tr = {'cuprate_CuO': int((idx_has('Cu', Ctr) & idx_has('O', Ctr)).sum()), 'Fe_pnictide_chalc': int((idx_has('Fe', Ctr) & (idx_has('As', Ctr) | idx_has('Se', Ctr) | idx_has('Te', Ctr) | idx_has('P', Ctr))).sum())}
    out['train_rows_by_family'] = fam_tr
    keyc = {}
    for i in range(n):
        keyc.setdefault(tuple(np.round(Cdv[i], 6)), []).append(i)
    dup = [ii for ii in keyc.values() if len(ii) > 1]
    if dup:
        sq = 0.0; cnt = 0; spread = []
        for ii in dup:
            yy = ydv[ii]; sq += float(((yy - yy.mean()) ** 2).sum()); cnt += len(ii); spread.append(float(yy.max() - yy.min()))
        out['dev_duplicate_compositions'] = {'dup_rows': cnt, 'dup_keys': len(dup), 'noise_floor_rmse_dup_rows': _r(math.sqrt(sq / cnt)), 'median_tc_range': _r(np.median(spread))}
    out['note'] = 'Controls use only element identity/composition and train labels; history rows are prior measured methods. Stratum RMSE differences are calculated; attributing them to chemical novelty vs within-system noise is interpretation.'
    s = json.dumps(out)
    if len(s.encode('utf-8')) > 8000:
        out.pop('rmse_by_subset_relation', None); s = json.dumps(out)
    if len(s.encode('utf-8')) > 8000:
        for k in list(out['rmse_by_maxJ_bin'].keys()):
            for pn in list(out['rmse_by_maxJ_bin'][k].keys()):
                if pn.startswith('ctl_jaccardknn') and pn != 'ctl_jaccardknn5':
                    out['rmse_by_maxJ_bin'][k].pop(pn, None)
        s = json.dumps(out)
    if len(s.encode('utf-8')) > 8000:
        out.pop('rmse_by_family', None)
    return out
