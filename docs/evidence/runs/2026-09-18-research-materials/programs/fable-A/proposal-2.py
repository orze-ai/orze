import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

def _hgb(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)

def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))

def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))

def _gmae(y, p, g):
    d = {}
    for yi, pi, gi in zip(y, p, g):
        d.setdefault(gi, []).append(abs(float(yi) - float(pi)))
    return float(np.mean([np.mean(v) for v in d.values()]))

def _metrics(y, p, g):
    return {'rmse': round(_rmse(y, p), 3), 'mae': round(_mae(y, p), 3), 'group_mae': round(_gmae(y, p, g), 3)}

def _family(C, elements):
    idx = {e: i for i, e in enumerate(elements)}
    cu = idx.get('Cu'); o = idx.get('O'); fe = idx.get('Fe')
    fam = []
    for row in C:
        if cu is not None and o is not None and row[cu] > 0 and row[o] > 0:
            fam.append('CuO')
        elif fe is not None and row[fe] > 0:
            fam.append('Fe')
        else:
            fam.append('other')
    return fam

def _strat(err, keys):
    d = {}
    for e, k in zip(err, keys):
        d.setdefault(str(k), []).append(float(e))
    res = {}
    for k, v in d.items():
        a = np.asarray(v)
        res[k] = {'n': int(a.size), 'rmse': round(float(np.sqrt(np.mean(a ** 2))), 2), 'mae': round(float(np.mean(a)), 2)}
    return res

def analyze(data, history, seed):
    tr = data['train']; dv = data['development']
    Xtr = np.asarray(tr['X'], float); Ctr = np.asarray(tr['C'], float); ytr = np.asarray(tr['y'], float); gtr = list(tr['groups'])
    Xdv = np.asarray(dv['X'], float); Cdv = np.asarray(dv['C'], float); ydv = np.asarray(dv['y'], float); gdv = list(dv['groups'])
    elements = list(tr['elements'])
    out = {'design': 'HGB settings identical to idea-baseline-2 (250 it, lr 0.08, 31 leaves, l2 1, seed 1729); representation is the only varied component; GroupKFold(3) on train by element set plus train->dev fits'}
    out['shapes'] = {'train_rows': int(len(ytr)), 'dev_rows': int(len(ydv)), 'train_systems': len(set(gtr)), 'dev_systems': len(set(gdv)), 'nX': int(Xtr.shape[1]), 'nC': int(Ctr.shape[1])}
    out['target'] = {'train_mean': round(float(ytr.mean()), 2), 'train_median': round(float(np.median(ytr)), 2), 'train_std': round(float(ytr.std()), 2), 'dev_mean': round(float(ydv.mean()), 2), 'dev_median': round(float(np.median(ydv)), 2), 'dev_std': round(float(ydv.std()), 2)}
    d = {}
    for i, r in enumerate(Ctr):
        d.setdefault(tuple(np.round(r, 5).tolist()), []).append(i)
    sq = []; ab = []; nrep = 0; rows_rep = 0
    for ix in d.values():
        if len(ix) > 1:
            nrep += 1; rows_rep += len(ix); yy = ytr[ix]; s = yy.sum(); m = len(ix)
            for j in range(m):
                r = yy[j] - (s - yy[j]) / (m - 1)
                sq.append(r * r); ab.append(abs(r))
    sq = np.asarray(sq)
    out['train_repeated_compositions'] = {'unique_compositions': len(d), 'compositions_with_repeats': nrep, 'rows_in_repeats': rows_rep, 'loo_rmse_within_repeats': round(float(np.sqrt(sq.mean())), 3) if sq.size else None, 'loo_mae_within_repeats': round(float(np.mean(ab)), 3) if ab else None, 'floor_rmse_all_rows_if_nonrepeats_perfect': round(float(np.sqrt(sq.sum() / len(ytr))), 3) if sq.size else None}
    ftr = _family(Ctr, elements); fdv = _family(Cdv, elements)
    fam = {}
    for name, f, y in (('train', ftr, ytr), ('dev', fdv, ydv)):
        fam[name] = {}
        for k in ('CuO', 'Fe', 'other'):
            m = np.asarray([x == k for x in f])
            fam[name][k] = {'frac_rows': round(float(m.mean()), 3), 'tc_mean': round(float(y[m].mean()), 2) if m.any() else None, 'tc_median': round(float(np.median(y[m])), 2) if m.any() else None}
    out['family_shift'] = fam
    sys_by_el = np.zeros(len(elements)); seen = set()
    for r, g in zip(Ctr, gtr):
        if g in seen:
            continue
        seen.add(g); sys_by_el += (r > 0)
    pres_dv = Cdv > 0
    min_cov = np.array([sys_by_el[p].min() if p.any() else 0.0 for p in pres_dv])
    cov_bin = np.where(min_cov == 0, 'absent', np.where(min_cov < 5, 'rare<5', np.where(min_cov < 30, 'mid<30', 'common')))
    out['dev_rows_by_min_train_system_count_of_elements'] = {k: int((cov_bin == k).sum()) for k in ('absent', 'rare<5', 'mid<30', 'common')}
    out['elements_absent_from_train_present_in_dev'] = [elements[i] for i in range(len(elements)) if sys_by_el[i] == 0 and pres_dv[:, i].any()]
    nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(Ctr)
    dist, ind = nn.kneighbors(Cdv)
    maxcos = 1.0 - dist[:, 0]
    knn_pred = ytr[ind].mean(axis=1); knn1 = ytr[ind[:, 0]]
    reps = {'X': (Xtr, Xdv), 'C': (Ctr, Cdv), 'XC': (np.hstack([Xtr, Ctr]), np.hstack([Xdv, Cdv]))}
    dev_res = {}; preds = {}
    for name in ('X', 'C', 'XC'):
        A, B = reps[name]
        p = _hgb(1729).fit(A, ytr).predict(B); preds[name] = p; dev_res[name] = _metrics(ydv, p, gdv)
    p = np.expm1(_hgb(1729).fit(reps['XC'][0], np.log1p(ytr)).predict(reps['XC'][1])); preds['XC_log1p'] = p; dev_res['XC_log1p'] = _metrics(ydv, p, gdv)
    dev_res['knn5_cosC'] = _metrics(ydv, knn_pred, gdv); dev_res['knn1_cosC'] = _metrics(ydv, knn1, gdv)
    dev_res['train_median'] = _metrics(ydv, np.full_like(ydv, np.median(ytr)), gdv)
    out['dev_by_representation'] = dev_res
    hist_check = None
    try:
        for h in history:
            if isinstance(h, dict) and h.get('action_id') == 'd81da712963c24b0e0581c92881e88ef6155ce721f33df1a06d438b0f13e3d90':
                hp = h.get('prediction')
                if hp is not None and len(hp) == len(ydv):
                    hp = np.asarray(hp, float)
                    hist_check = {'hist_pred_rmse': round(_rmse(ydv, hp), 3), 'refit_vs_hist_pred_rmse': round(_rmse(hp, preds['X']), 3)}
    except Exception as ex:
        hist_check = {'error': str(ex)[:100]}
    out['baseline2_history_prediction_check'] = hist_check
    err = np.abs(ydv - preds['X'])
    tcl = np.array(['<5', '5-20', '20-50', '50-90', '>=90'])[np.digitize(ydv, [5, 20, 50, 90])]
    cosl = np.array(['<0.9', '0.9-0.99', '0.99-0.9999', '>=0.9999'])[np.digitize(maxcos, [0.9, 0.99, 0.9999])]
    out['hgbX_dev_err_by_tc_bin'] = _strat(err, tcl)
    out['hgbX_dev_err_by_family'] = _strat(err, fdv)
    out['hgbX_dev_err_by_max_cosC_to_train'] = _strat(err, cosl)
    out['hgbX_dev_err_by_element_coverage'] = _strat(err, cov_bin)
    sb = {}
    for k in np.unique(tcl):
        m = tcl == k; sb[str(k)] = round(float((preds['X'][m] - ydv[m]).mean()), 2)
    out['hgbX_dev_signed_bias_pred_minus_true_by_tc_bin'] = sb
    se = np.sort(err ** 2)[::-1]; k = max(1, int(0.05 * len(se)))
    out['hgbX_dev_sse_share_top5pct_rows'] = round(float(se[:k].sum() / se.sum()), 3)
    sysd = {}
    for g, e, y in zip(gdv, err, ydv):
        sysd.setdefault(g, []).append((float(e), float(y)))
    worst = sorted(((float(np.mean([a for a, _ in v])), g, len(v), float(np.mean([b for _, b in v]))) for g, v in sysd.items()), reverse=True)[:8]
    out['hgbX_worst_dev_systems'] = [{'system': g, 'n': n, 'mae': round(m, 1), 'tc_mean': round(t, 1)} for m, g, n, t in worst]
    gkf = GroupKFold(n_splits=3)
    cv = {}
    for name in ('X', 'C', 'XC', 'XC_log1p'):
        A = reps['XC'][0] if name.startswith('XC') else reps[name][0]
        oof = np.zeros_like(ytr)
        for tri, tei in gkf.split(A, ytr, gtr):
            if name == 'XC_log1p':
                oof[tei] = np.expm1(_hgb(1729).fit(A[tri], np.log1p(ytr[tri])).predict(A[tei]))
            else:
                oof[tei] = _hgb(1729).fit(A[tri], ytr[tri]).predict(A[tei])
        cv[name] = _metrics(ytr, oof, gtr)
    out['train_groupkfold3_oof_by_representation'] = cv
    out['note'] = 'All values above are calculations on supplied rows; family labels are C-derived heuristics, not physical classes; interpretation deferred to the next hypothesis.'
    return out
