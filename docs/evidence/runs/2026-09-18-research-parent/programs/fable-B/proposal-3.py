import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

LOGF = ['jitter(%)', 'jitter(abs)', 'jitter:rap', 'jitter:ppq5', 'jitter:ddp', 'shimmer', 'shimmer(db)', 'shimmer:apq3', 'shimmer:apq5', 'shimmer:apq11', 'shimmer:dda', 'nhr']


def fit_predict(train, inputs, seed):
    fn = list(train['feature_names'])
    X = np.asarray(train['X'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    g = np.asarray([str(v) for v in train['groups']])
    Xi = np.asarray(inputs['X'], dtype=float)
    idx = {n: i for i, n in enumerate(fn)}
    ia, isx, it = idx['age'], idx['sex'], idx['test_time']
    voice = [i for i, n in enumerate(fn) if n not in ('age', 'sex', 'test_time')]
    floors = {}
    for nm in LOGF:
        j = idx[nm]
        pos = X[:, j][X[:, j] > 0]
        floors[nm] = float(pos.min()) if pos.size else 1e-6

    def tf(M):
        M = np.array(M, dtype=float, copy=True)
        for nm in LOGF:
            j = idx[nm]
            M[:, j] = np.log(np.maximum(M[:, j], floors[nm]))
        return M

    Xt = tf(X)
    Xit = tf(Xi)

    def ak(M):
        return np.asarray(['%.0f_%.0f' % (a, s) for a, s in zip(M[:, ia], M[:, isx])])

    akey_tr = ak(X)
    akey_in = ak(Xi)
    bsets = {'age': [ia], 'demo': [ia, isx], 'demo_voice': [ia, isx] + voice, 'voice': voice}
    wsets = {'time': [it], 'time_voice': [it] + voice, 'voice': voice}
    balphas = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    walphas = [1.0, 10.0, 100.0, 1000.0, 10000.0]

    def center_by(M, keys):
        C = np.zeros_like(M)
        for k in set(keys.tolist()):
            m = keys == k
            C[m] = M[m] - M[m].mean(axis=0)
        return C

    def std_fit(A):
        m = A.mean(axis=0)
        s = A.std(axis=0)
        return m, np.where(s < 1e-12, 1.0, s)

    def run(trm, Xte, keys, bspec, wspec, want_models=False):
        gt = g[trm]
        Xtr = Xt[trm]
        ytr = y[trm]
        pts = sorted(set(gt.tolist()))
        Pf = np.array([Xtr[gt == p].mean(axis=0) for p in pts])
        Pyf = np.array([ytr[gt == p].mean() for p in pts])
        medf = float(np.median(ytr))
        uk = sorted(set(keys.tolist()))
        Kf = np.array([Xte[keys == k].mean(axis=0) for k in uk])
        info = {}
        if bspec[0] == 'const':
            bk = np.full(len(uk), medf)
        else:
            cols = bsets[bspec[0]]
            A = Pf[:, cols]
            m, s = std_fit(A)
            rb = Ridge(alpha=bspec[1]).fit((A - m) / s, Pyf)
            bk = rb.predict((Kf[:, cols] - m) / s)
            if want_models:
                info['between_coef_std'] = {fn[c]: round(float(v), 3) for c, v in zip(cols, rb.coef_)}
                info['between_intercept'] = round(float(rb.intercept_), 3)
        kpos = {k: i for i, k in enumerate(uk)}
        bp = np.array([bk[kpos[k]] for k in keys])
        wp = np.zeros(len(keys))
        if wspec[0] != 'none':
            cols = wsets[wspec[0]]
            Xc = center_by(Xtr, gt)
            pmd = dict(zip(pts, Pyf))
            rr = ytr - np.array([pmd[p] for p in gt])
            A = Xc[:, cols]
            s = A.std(axis=0)
            s = np.where(s < 1e-12, 1.0, s)
            rw = Ridge(alpha=wspec[1], fit_intercept=False).fit(A / s, rr)
            wp = rw.predict(center_by(Xte, keys)[:, cols] / s)
            if want_models:
                info['within_coef_std'] = {fn[c]: round(float(v), 3) for c, v in zip(cols, rw.coef_)}
                if it in cols:
                    q = cols.index(it)
                    info['within_time_slope_per_day'] = round(float(rw.coef_[q] / s[q]), 5)
        pred = np.clip(bp + wp, ytr.min(), ytr.max())
        return pred, bp, wp, info

    def metrics(pred, yt, gt):
        e = pred - yt
        gm = float(np.mean([np.abs(e[gt == p]).mean() for p in set(gt.tolist())]))
        return float(np.sqrt(np.mean(e ** 2))), gm

    folds = list(GroupKFold(n_splits=7).split(X, y, g))

    def cv(bspec, wspec, use_true):
        ps = np.zeros(len(y))
        for tr, te in folds:
            trm = np.zeros(len(y), dtype=bool)
            trm[tr] = True
            keys = g[te] if use_true else akey_tr[te]
            ps[te] = run(trm, Xt[te], keys, bspec, wspec)[0]
        return metrics(ps, y, g)

    def row(spec, r, gm):
        return {'set': spec[0], 'alpha': spec[1], 'cv_rmse': round(r, 4), 'cv_gmae': round(gm, 4)}

    bcands = [('const', None)] + [(s, a) for s in bsets for a in balphas]
    btab = [row(b, *cv(b, ('none', None), True)) for b in bcands]
    bbest = min(btab, key=lambda d: d['cv_rmse'])
    bspec = (bbest['set'], bbest['alpha'])
    wcands = [('none', None)] + [(s, a) for s in wsets for a in walphas]
    wtab = [row(w, *cv(bspec, w, True)) for w in wcands]
    wbest = min(wtab, key=lambda d: d['cv_rmse'])
    wspec = (wbest['set'], wbest['alpha'])
    cv_true = cv(bspec, wspec, True)
    cv_ak = cv(bspec, wspec, False)
    med_cv = cv(('const', None), ('none', None), True)

    allm = np.ones(len(y), dtype=bool)
    pred, bp, wp, info = run(allm, Xit, akey_in, bspec, wspec, want_models=True)
    pats = sorted(set(g.tolist()))
    Py = np.array([y[g == p].mean() for p in pats])
    within_sd = float(np.sqrt(np.mean((y - np.array([Py[pats.index(p)] for p in g])) ** 2)))
    ukin, cnt = np.unique(akey_in, return_counts=True)
    findings = {
        'design': 'two-component model: between = ridge on patient-mean (log-voice, age, sex) features fit on train patients, applied to pseudo-patient means; within = intercept-free ridge on features centered per (pseudo-)patient fit on train residuals from patient mean y',
        'transductive_use_of_inputs': 'yes: evaluation rows with identical (age,sex) are grouped into pseudo-patients; their unlabeled feature means and per-group centering are used. No labels, groups or row ids. Log floors, scaling and clipping from train only.',
        'fallback_used': False,
        'train_patients': len(pats),
        'train_distinct_agesex_keys': int(len(set(akey_tr.tolist()))),
        'input_rows': int(len(Xi)),
        'input_pseudo_patients': int(len(ukin)),
        'input_pseudo_patient_row_counts': sorted([int(c) for c in cnt], reverse=True),
        'train_between_patient_std_of_means': round(float(Py.std()), 3),
        'train_within_patient_pooled_rmsd': round(within_sd, 3),
        'selection_rule': 'stage1: between model by min 7-fold GroupKFold row RMSE over train patients with within=none; stage2: within model by same CV given chosen between; CV uses true patient groups as keys',
        'median_cv': {'cv_rmse': round(med_cv[0], 4), 'cv_gmae': round(med_cv[1], 4)},
        'between_table_top10': sorted(btab, key=lambda d: d['cv_rmse'])[:10],
        'between_const': btab[0],
        'within_table': wtab,
        'chosen': {'between': bbest, 'within': wbest},
        'cv_chosen_true_patient_keys': {'cv_rmse': round(cv_true[0], 4), 'cv_gmae': round(cv_true[1], 4)},
        'cv_chosen_agesex_pseudo_keys': {'cv_rmse': round(cv_ak[0], 4), 'cv_gmae': round(cv_ak[1], 4)},
        'model': info,
        'train_median': round(float(np.median(y)), 3),
        'pred_summary': {'mean': round(float(pred.mean()), 3), 'std': round(float(pred.std()), 3), 'min': round(float(pred.min()), 3), 'max': round(float(pred.max()), 3)},
        'between_component_std': round(float(bp.std()), 3),
        'within_component_std': round(float(wp.std()), 3),
    }
    return {'prediction': [float(v) for v in pred], 'findings': findings}
