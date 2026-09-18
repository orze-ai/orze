import json
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge


def _arr(d):
    X = np.asarray(d['X'], dtype=float)
    y = np.asarray(d['y'], dtype=float)
    g = np.asarray([str(v) for v in d['groups']])
    return X, y, g


def _psum(X, y, g, ids):
    M = np.array([X[g == p].mean(axis=0) for p in ids])
    L = np.array([float(y[g == p].mean()) for p in ids])
    N = np.array([int((g == p).sum()) for p in ids])
    return M, L, N


def _rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _r(x, k=4):
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return round(v, k)


def _corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or np.std(a) <= 0 or np.std(b) <= 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _lopo_ridge(M, L, cols, alpha):
    n = len(L)
    pred = np.empty(n)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        A = M[tr][:, cols]
        b = L[tr]
        mu = A.mean(axis=0)
        sd = A.std(axis=0)
        sd[sd == 0] = 1.0
        mdl = Ridge(alpha=alpha).fit((A - mu) / sd, b)
        pred[i] = float(mdl.predict((M[i:i + 1, cols] - mu) / sd)[0])
    return pred


def _lopo_const(L):
    n = len(L)
    return np.array([float(np.mean(np.delete(L, i))) for i in range(n)])


def _center(X, y, g, ids):
    Xc = X.copy()
    yc = y.copy()
    for p in ids:
        m = g == p
        Xc[m] = Xc[m] - X[m].mean(axis=0)
        yc[m] = yc[m] - y[m].mean()
    return Xc, yc


def analyze(data, history, seed):
    rng = np.random.default_rng(int(seed) if seed is not None else 1729)
    out = {
        'design': 'diagnostics: variance decomposition, 35-patient between-patient LOPO ridge + permutation, within-patient centred ridge trained on train and tested on held-out dev patients, constant-choice bootstrap over 7 dev patients, (age,sex) pseudo-patient uniqueness, replay of historical dev predictions',
        'fallback_used': False,
        'errors': [],
        'caveat': 'development labels used here for diagnosis only; any dev-fitted number is at risk of not transferring to the 7 confirmation patients after refit on train+development'
    }
    try:
        Xtr, ytr, gtr = _arr(data['train'])
        Xdv, ydv, gdv = _arr(data['development'])
        fn = [str(s) for s in data['train']['feature_names']]
    except Exception as e:
        out['fallback_used'] = True
        out['errors'].append('load:' + repr(e)[:120])
        return out
    ptr = sorted(set(gtr.tolist()))
    pdv = sorted(set(gdv.tolist()))
    Mtr, Ltr, Ntr = _psum(Xtr, ytr, gtr, ptr)
    Mdv, Ldv, Ndv = _psum(Xdv, ydv, gdv, pdv)
    idx = {}
    for i, nm in enumerate(fn):
        idx[nm] = i
    i_age = idx.get('age', 0)
    i_sex = idx.get('sex', 1)
    i_tt = idx.get('test_time', 2)
    p = Xtr.shape[1]
    out['n'] = {'train_rows': int(len(ytr)), 'train_patients': len(ptr),
                'dev_rows': int(len(ydv)), 'dev_patients': len(pdv),
                'n_features': int(p), 'rows_per_patient_dev': [int(v) for v in Ndv]}

    # 1) variance decomposition / oracle floors
    try:
        wtr = ytr - np.concatenate([np.full(int((gtr == q).sum()), float(ytr[gtr == q].mean())) for q in ptr])
        # rebuild aligned within-residuals safely
        wtr = np.empty_like(ytr)
        for q in ptr:
            m = gtr == q
            wtr[m] = ytr[m] - ytr[m].mean()
        wdv = np.empty_like(ydv)
        for q in pdv:
            m = gdv == q
            wdv[m] = ydv[m] - ydv[m].mean()
        out['variance'] = {
            'train_total_sd': _r(np.std(ytr)),
            'train_within_rmse_oracle_patient_const': _r(np.sqrt(np.mean(wtr ** 2))),
            'train_between_patient_level_sd': _r(np.std(Ltr)),
            'dev_total_sd': _r(np.std(ydv)),
            'dev_within_rmse_oracle_patient_const': _r(np.sqrt(np.mean(wdv ** 2))),
            'dev_between_patient_level_sd': _r(np.std(Ldv)),
            'dev_within_share_of_total_var': _r(np.mean(wdv ** 2) / max(np.var(ydv), 1e-12))
        }
    except Exception as e:
        out['errors'].append('variance:' + repr(e)[:120])

    # 2) constant candidates measured on dev + bootstrap over dev patients
    try:
        cands = {'train_row_median': float(np.median(ytr)),
                 'train_row_mean': float(np.mean(ytr)),
                 'train_pat_mean_mean': float(np.mean(Ltr)),
                 'train_pat_mean_median': float(np.median(Ltr))}
        cres = {}
        for k, c in cands.items():
            pm = [float(np.mean(np.abs(ydv[gdv == q] - c))) for q in pdv]
            cres[k] = {'c': _r(c), 'dev_rmse': _r(_rmse(ydv, np.full(len(ydv), c))),
                       'dev_group_mae': _r(np.mean(pm))}
        grid = np.linspace(float(np.min(ydv)), float(np.max(ydv)), 601)
        gr = [_rmse(ydv, np.full(len(ydv), c)) for c in grid]
        j = int(np.argmin(gr))
        cres['dev_oracle_constant'] = {'c': _r(grid[j]), 'dev_rmse': _r(gr[j])}
        out['constants'] = cres
        blocks = [ydv[gdv == q] for q in pdv]
        cA = cands['train_row_median']
        cB = cands['train_row_mean']
        dif = np.empty(3000)
        for b in range(3000):
            sel = rng.integers(0, len(blocks), len(blocks))
            yy = np.concatenate([blocks[int(s)] for s in sel])
            dif[b] = _rmse(yy, np.full(len(yy), cA)) - _rmse(yy, np.full(len(yy), cB))
        out['constant_bootstrap_median_minus_mean'] = {
            'point_dev_rmse_diff': _r(_rmse(ydv, np.full(len(ydv), cA)) - _rmse(ydv, np.full(len(ydv), cB))),
            'p2_5': _r(np.percentile(dif, 2.5)), 'p97_5': _r(np.percentile(dif, 97.5)),
            'frac_favouring_median': _r(float(np.mean(dif < 0))), 'n_boot': 3000,
            'unit': 'patient-level bootstrap over 7 dev patients'}
    except Exception as e:
        out['errors'].append('constants:' + repr(e)[:120])

    # 3) between-patient mapping, 35-patient LOPO + permutation
    try:
        Mall = np.vstack([Mtr, Mdv])
        Lall = np.concatenate([Ltr, Ldv])
        allc = list(range(p))
        demo = [c for c in [i_age, i_sex] if c < p]
        voice = [c for c in range(p) if c not in (i_age, i_sex, i_tt)]
        base = _rmse(Lall, _lopo_const(Lall))
        res = {}
        best = (None, 1e18)
        for name, cols in [('all19', allc), ('age_sex', demo), ('voice16', voice)]:
            row = {}
            for a in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
                v = _rmse(Lall, _lopo_ridge(Mall, Lall, cols, a))
                row[str(int(a))] = _r(v)
                if v < best[1]:
                    best = ((name, cols, a), v)
            res[name] = row
        out['between_patient_lopo35'] = {'constant_only_rmse': _r(base), 'ridge_rmse_by_alpha': res,
                                         'best': {'cols': best[0][0], 'alpha': best[0][2], 'rmse': _r(best[1])}}
        cols = best[0][1]
        a = best[0][2]
        obs = best[1]
        nperm = 300
        cnt = 0
        for b in range(nperm):
            Lp = rng.permutation(Lall)
            if _rmse(Lp, _lopo_ridge(Mall, Lp, cols, a)) <= obs:
                cnt += 1
        out['between_patient_lopo35']['permutation'] = {
            'n_perm': nperm, 'p_value': _r((cnt + 1.0) / (nperm + 1.0)),
            'note': 'alpha/cols fixed at observed best, so p is optimistic-biased toward significance'}
        # sign replication of patient-level correlations across splits
        ct = np.array([_corr(Mtr[:, c], Ltr) for c in range(p)])
        cd = np.array([_corr(Mdv[:, c], Ldv) for c in range(p)])
        order = np.argsort(-np.abs(ct))[:8]
        out['patient_level_corr_top8'] = [[fn[c][:22], _r(ct[c], 3), _r(cd[c], 3)] for c in order]
        out['patient_level_corr_sign_agreement_all'] = _r(float(np.mean(np.sign(ct) == np.sign(cd))), 3)
    except Exception as e:
        out['errors'].append('between:' + repr(e)[:120])

    # 4) within-patient signal: fit on train centred rows, test on dev centred rows
    try:
        Xtc, ytc = _center(Xtr, ytr, gtr, ptr)
        Xdc, ydc = _center(Xdv, ydv, gdv, pdv)
        sd = Xtc.std(axis=0)
        sd[sd == 0] = 1.0
        wr = {}
        for a in [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]:
            mdl = Ridge(alpha=a, fit_intercept=False).fit(Xtc / sd, ytc)
            pr = mdl.predict(Xdc / sd)
            wr[str(int(a))] = {'dev_within_rmse': _r(_rmse(ydc, pr)),
                               'corr': _r(_corr(pr, ydc), 3),
                               'shrink_opt': _r(float(np.dot(pr, ydc) / max(np.dot(pr, pr), 1e-12)), 3)}
        out['within_patient_ridge'] = {'zero_pred_dev_within_rmse': _r(_rmse(ydc, np.zeros_like(ydc))),
                                       'by_alpha': wr,
                                       'note': 'oracle patient centring used only to isolate the within-patient component'}
        # pooled within-patient test_time slope
        num = float(np.dot(Xtc[:, i_tt], ytc))
        den = float(np.dot(Xtc[:, i_tt], Xtc[:, i_tt]))
        slope = num / den if den > 0 else 0.0
        prt = slope * Xdc[:, i_tt]
        sl = []
        for q in ptr:
            m = gtr == q
            v = Xtr[m, i_tt] - Xtr[m, i_tt].mean()
            d = float(np.dot(v, v))
            if d > 0:
                sl.append(float(np.dot(v, ytr[m] - ytr[m].mean()) / d))
        sl = np.array(sl) if len(sl) else np.zeros(1)
        out['within_test_time'] = {'pooled_slope_upd_per_day': _r(slope, 5),
                                   'per_patient_slope_mean': _r(np.mean(sl), 5),
                                   'per_patient_slope_sd': _r(np.std(sl), 5),
                                   'frac_positive': _r(float(np.mean(sl > 0)), 3),
                                   'dev_within_rmse_with_slope': _r(_rmse(ydc, prt)),
                                   'dev_within_rmse_zero': _r(_rmse(ydc, np.zeros_like(ydc)))}
        ctw = np.array([_corr(Xtc[:, c], ytc) for c in range(p)])
        cdw = np.array([_corr(Xdc[:, c], ydc) for c in range(p)])
        ow = np.argsort(-np.abs(ctw))[:8]
        out['within_corr_top8'] = [[fn[c][:22], _r(ctw[c], 3), _r(cdw[c], 3)] for c in ow]
        out['within_corr_sign_agreement_all'] = _r(float(np.mean(np.sign(ctw) == np.sign(cdw))), 3)
    except Exception as e:
        out['errors'].append('within:' + repr(e)[:120])

    # 5) (age,sex) pseudo-patient key uniqueness
    try:
        def keys(M, ids):
            d = defaultdict(list)
            for i, q in enumerate(ids):
                d[(round(float(M[i, i_age]), 6), round(float(M[i, i_sex]), 6))].append(q)
            return d
        ktr = keys(Mtr, ptr)
        kdv = keys(Mdv, pdv)
        kal = keys(np.vstack([Mtr, Mdv]), list(ptr) + list(pdv))
        out['pseudo_patient_key'] = {
            'train_patients': len(ptr), 'train_distinct_keys': len(ktr),
            'train_max_collision': int(max(len(v) for v in ktr.values())),
            'dev_patients': len(pdv), 'dev_distinct_keys': len(kdv),
            'dev_max_collision': int(max(len(v) for v in kdv.values())),
            'combined35_distinct_keys': len(kal),
            'combined35_max_collision': int(max(len(v) for v in kal.values())),
            'implication': 'collision rate bounds reliability of transductive (age,sex) grouping on unseen confirmation patients'}
    except Exception as e:
        out['errors'].append('key:' + repr(e)[:120])

    # 6) replay historical dev predictions against dev labels
    try:
        rid = {str(v): i for i, v in enumerate(data['development']['row_ids'])}
        hh = []
        for h in (history or []):
            try:
                pr = h.get('prediction')
                if not pr:
                    continue
                ids = [str(v) for v in pr.get('row_ids', [])]
                vals = np.asarray(pr.get('prediction', []), dtype=float)
                if len(ids) != len(vals) or len(ids) == 0:
                    continue
                pos = np.array([rid.get(s, -1) for s in ids])
                ok = pos >= 0
                if ok.sum() < len(ids):
                    continue
                yy = ydv[pos]
                gg = gdv[pos]
                per = sorted([(str(q), _r(np.sqrt(np.mean((yy[gg == q] - vals[gg == q]) ** 2)))) for q in pdv],
                             key=lambda t: (t[1] is None, -(t[1] or 0)))[:3]
                hh.append({'task_id': str(h.get('task_id'))[:24],
                           'valid': bool(h.get('valid')),
                           'recomputed_dev_rmse': _r(_rmse(yy, vals)),
                           'pred_sd': _r(np.std(vals)), 'pred_mean': _r(np.mean(vals)),
                           'worst3_patient_rmse': per})
            except Exception:
                continue
        out['history_replay'] = hh[:8]
    except Exception as e:
        out['errors'].append('history:' + repr(e)[:120])

    for k in ['history_replay', 'within_corr_top8', 'patient_level_corr_top8', 'between_patient_lopo35']:
        try:
            if len(json.dumps(out).encode('utf-8')) <= 8000:
                break
            out.pop(k, None)
            out['errors'].append('dropped_for_size:' + k)
        except Exception:
            break
    return out
