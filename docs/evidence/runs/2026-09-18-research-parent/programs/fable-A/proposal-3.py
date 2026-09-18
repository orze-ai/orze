import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

LOG_KEYS = ('jitter', 'shimmer', 'nhr')

def col_index(names):
    idx = {}
    for j, nm in enumerate(names):
        idx[str(nm).lower()] = j
    age = idx.get('age', 0)
    sex = idx.get('sex', 1)
    tt = idx.get('test_time', 2)
    voice = [j for j in range(len(names)) if j not in (age, sex, tt)]
    return age, sex, tt, voice

def transform_rows(X, names):
    Z = np.array(X, dtype=float, copy=True)
    for j, nm in enumerate(names):
        low = str(nm).lower()
        if any(k in low for k in LOG_KEYS):
            Z[:, j] = np.log1p(np.clip(Z[:, j], 0.0, None))
    return Z

def pseudo_keys(Z, age, sex):
    return [(int(round(Z[i, age])), int(round(Z[i, sex]))) for i in range(Z.shape[0])]

def aggregate(Z, keys, age, sex, tt, voice):
    order = {}
    for i, k in enumerate(keys):
        order.setdefault(k, []).append(i)
    out = {}
    for k in order:
        r = np.array(order[k], dtype=int)
        V = Z[r][:, voice]
        m = V.mean(axis=0)
        s = V.std(axis=0) if len(r) > 1 else np.zeros(len(voice))
        out[k] = {'mean': m, 'std': s, 'age': float(Z[r, age].mean()), 'sex': float(Z[r, sex].mean()), 't': float(Z[r, tt].mean()), 'rows': r, 'n': int(len(r))}
    return out

def build_matrix(aggs, order, spec):
    rows = []
    for k in order:
        a = aggs[k]
        parts = []
        if 'demo' in spec:
            parts.append(np.array([a['age'], a['sex']], dtype=float))
        if 'mean' in spec:
            parts.append(a['mean'])
        if 'std' in spec:
            parts.append(a['std'])
        rows.append(np.concatenate(parts))
    return np.vstack(rows)

def fit_agg_model(A, y, kind, alpha):
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    S = (A - mu) / sd
    yy = np.asarray(y, dtype=float)
    if kind == 'const':
        c = float(yy.mean())
        return lambda Q: np.full(Q.shape[0], c)
    if kind == 'ridge':
        m = Ridge(alpha=float(alpha)).fit(S, yy)
        return lambda Q: m.predict((Q - mu) / sd)
    if kind == 'knn':
        k = int(alpha)
        def pred(Q):
            Qs = (Q - mu) / sd
            D = ((Qs[:, None, :] - S[None, :, :]) ** 2).sum(axis=2)
            nn = np.argsort(D, axis=1)[:, :k]
            return yy[nn].mean(axis=1)
        return pred
    raise ValueError('unknown kind ' + str(kind))

def pooled_slope(Z, y, groups, tt):
    num = 0.0
    den = 0.0
    order = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    for g in order:
        r = np.array(order[g], dtype=int)
        if len(r) < 2:
            continue
        tc = Z[r, tt] - Z[r, tt].mean()
        yc = y[r] - y[r].mean()
        num += float((tc * yc).sum())
        den += float((tc * tc).sum())
    return num / den if den > 0 else 0.0

def fit_candidate(cand, Ztr, ytr, gtr, age, sex, tt, voice):
    kind, alpha, spec, use_slope = cand
    slope = pooled_slope(Ztr, ytr, list(gtr), tt) if use_slope else 0.0
    if kind == 'rowridge':
        mu = Ztr.mean(axis=0)
        sd = Ztr.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        m = Ridge(alpha=float(alpha)).fit((Ztr - mu) / sd, ytr)
        return {'row': (m, mu, sd), 'slope': slope}
    aggs = aggregate(Ztr, list(gtr), age, sex, tt, voice)
    order = list(aggs.keys())
    A = build_matrix(aggs, order, spec)
    ymean = np.array([float(ytr[aggs[k]['rows']].mean()) for k in order])
    f = fit_agg_model(A, ymean, kind, alpha)
    return {'agg': f, 'spec': spec, 'slope': slope}

def predict_candidate(model, Zev, age, sex, tt, voice):
    if 'row' in model:
        m, mu, sd = model['row']
        return m.predict((Zev - mu) / sd), 0, []
    keys = pseudo_keys(Zev, age, sex)
    aggs = aggregate(Zev, keys, age, sex, tt, voice)
    order = list(aggs.keys())
    A = build_matrix(aggs, order, model['spec'])
    gp = model['agg'](A)
    pred = np.zeros(Zev.shape[0])
    for gi, k in enumerate(order):
        a = aggs[k]
        pred[a['rows']] = gp[gi] + model['slope'] * (Zev[a['rows'], tt] - a['t'])
    return pred, len(order), [aggs[k]['n'] for k in order]

def fit_predict(train, inputs, seed):
    names = list(train['feature_names'])
    Xtr = np.asarray(train['X'], dtype=float)
    ytr = np.asarray(train['y'], dtype=float)
    gtr = np.asarray(train['groups'])
    Xev = np.asarray(inputs['X'], dtype=float)
    ymean_train = float(ytr.mean())
    findings = {'method': 'patient-level aggregate voice representation (mean/std of log jitter/shimmer/NHR and raw other voice measures) trained on train patient means; unlabeled inputs pseudo-grouped by rounded (age, sex) key; optional pooled within-patient test_time slope; row-level ridge a1000 as contrast under identical folds', 'fallback_used_train_mean': False, 'error': None, 'transductive_use': 'yes: evaluation rows sharing a rounded (age, sex) key are pooled to compute voice aggregates and mean test_time; no labels, groups or row ids are used; all scaling, aggregation statistics and selection come from train rows only'}
    pred = np.full(Xev.shape[0], ymean_train)
    try:
        age, sex, tt, voice = col_index(names)
        Ztr = transform_rows(Xtr, names)
        Zev = transform_rows(Xev, names)
        cands = [
            ('const', 0, 'mean', False), ('const', 0, 'mean', True),
            ('ridge', 10, 'demo', False), ('ridge', 10, 'demo', True),
            ('ridge', 100, 'mean', False), ('ridge', 100, 'mean', True),
            ('ridge', 1000, 'mean', False), ('ridge', 1000, 'mean', True),
            ('ridge', 100, 'demo+mean', False), ('ridge', 1000, 'demo+mean', False), ('ridge', 1000, 'demo+mean', True),
            ('ridge', 1000, 'demo+mean+std', False), ('ridge', 1000, 'demo+mean+std', True),
            ('knn', 5, 'mean', False), ('knn', 5, 'mean', True),
            ('knn', 7, 'demo+mean', False),
            ('rowridge', 1000, 'all', False),
        ]
        gkf = GroupKFold(n_splits=7)
        results = []
        for cand in cands:
            kind, alpha, spec, use_slope = cand
            name = kind + '_' + spec + '_a' + str(alpha) + ('_slope' if use_slope else '')
            errs = []
            gm = []
            gr = []
            npg = []
            ok = True
            for tri, tei in gkf.split(Ztr, ytr, gtr):
                model = fit_candidate(cand, Ztr[tri], ytr[tri], gtr[tri], age, sex, tt, voice)
                p, ng, sizes = predict_candidate(model, Ztr[tei], age, sex, tt, voice)
                if not np.all(np.isfinite(p)):
                    ok = False
                    break
                e = p - ytr[tei]
                errs.append(e)
                npg.append(int(ng))
                gte = gtr[tei]
                for g in list(dict.fromkeys(gte.tolist())):
                    mask = gte == g
                    gm.append(float(np.abs(e[mask]).mean()))
                    gr.append(float(e[mask].mean()))
            rec = {'name': name, 'ok': ok}
            if ok:
                E = np.concatenate(errs)
                rec['row_rmse'] = round(float(np.sqrt((E ** 2).mean())), 4)
                rec['row_mae'] = round(float(np.abs(E).mean()), 4)
                rec['group_mae'] = round(float(np.mean(gm)), 4)
                rec['group_bias_rmse'] = round(float(np.sqrt(np.mean(np.square(gr)))), 4)
                rec['pseudo_groups_per_fold'] = npg
            results.append(rec)
        okres = [r for r in results if r['ok']]
        best = min(okres, key=lambda r: r['row_rmse'])
        bi = [r['name'] for r in results].index(best['name'])
        cand = cands[bi]
        model = fit_candidate(cand, Ztr, ytr, gtr, age, sex, tt, voice)
        p, ng, sizes = predict_candidate(model, Zev, age, sex, tt, voice)
        lo = float(ytr.min()) - 5.0
        hi = float(ytr.max()) + 5.0
        bad = ~np.isfinite(p)
        p = np.where(bad, ymean_train, p)
        p = np.clip(p, lo, hi)
        pred = p
        tk = pseudo_keys(Ztr, age, sex)
        kmap = {}
        for i, k in enumerate(tk):
            kmap.setdefault(k, set()).add(str(gtr[i]))
        findings.update({'chosen': best['name'], 'selection_rule': 'lowest 7-fold GroupKFold row RMSE on train patients, held-out folds pseudo-grouped by rounded (age,sex) exactly as evaluation inputs; ties earliest in candidate order', 'chosen_slope_per_test_time_unit': round(float(model['slope']), 6), 'candidates': results, 'n_train_patients': int(len(set(gtr.tolist()))), 'n_train_rows': int(Xtr.shape[0]), 'n_train_pseudo_keys': int(len(kmap)), 'n_train_keys_merging_multiple_patients': int(sum(1 for k in kmap if len(kmap[k]) > 1)), 'n_eval_rows': int(Xev.shape[0]), 'n_eval_pseudo_groups': int(ng), 'eval_pseudo_group_sizes': sorted([int(s) for s in sizes]), 'n_nonfinite_replaced': int(bad.sum()), 'clip_range': [round(lo, 3), round(hi, 3)], 'train_y_mean': round(ymean_train, 4), 'note': 'row RMSE in candidates is patient-held-out on train; no evaluation labels used; pseudo-groups can merge distinct patients with equal rounded age and sex'})
    except Exception as e:
        findings['fallback_used_train_mean'] = True
        findings['error'] = str(e)[:500]
        pred = np.full(Xev.shape[0], ymean_train)
    return {'prediction': [float(v) for v in pred], 'findings': findings}
