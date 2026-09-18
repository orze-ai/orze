import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

LOGK = ('jitter', 'shimmer', 'nhr')
ALPHAS = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def design(X, names):
    X = np.asarray(X, dtype=float)
    cols = [X[:, j] for j in range(X.shape[1])]
    cn = [str(n) for n in names]
    for j, n in enumerate(names):
        if any(k in str(n).lower() for k in LOGK):
            cols.append(np.log(np.maximum(X[:, j], 1e-8)))
            cn.append('log_' + str(n))
    return np.column_stack(cols), cn


def agg(F, y, g):
    y = np.asarray(y, dtype=float)
    gs = [str(v) for v in g]
    keys = sorted(set(gs))
    M = np.zeros((len(keys), F.shape[1]))
    V = np.zeros((len(keys), F.shape[1]))
    yb = np.zeros(len(keys))
    for i, k in enumerate(keys):
        m = np.array([s == k for s in gs])
        M[i] = F[m].mean(axis=0)
        V[i] = F[m].var(axis=0) if int(m.sum()) > 1 else 0.0
        yb[i] = y[m].mean()
    return keys, M, V, yb


def ridge(Z, t, a):
    p = Z.shape[1]
    return np.linalg.solve(Z.T @ Z + a * np.eye(p), Z.T @ t)


def fit_model(F, y, g):
    y = np.asarray(y, dtype=float)
    keys, M, V, yb = agg(F, y, g)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (M - mu) / sd
    n = len(keys)
    y0 = float(yb.mean())
    lopo = {}
    best = (ALPHAS[-1], float('inf'))
    for a in ALPHAS:
        se = 0.0
        for i in range(n):
            tr = [j for j in range(n) if j != i]
            m0 = float(yb[tr].mean())
            b = ridge(Z[tr], yb[tr] - m0, a)
            se += (m0 + float(Z[i] @ b) - yb[i]) ** 2
        r = float(np.sqrt(se / n))
        lopo['alpha_%g' % a] = round(r, 4)
        if r < best[1]:
            best = (a, r)
    se = 0.0
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        se += (float(yb[tr].mean()) - yb[i]) ** 2
    lopo['const'] = round(float(np.sqrt(se / n)), 4)
    b = ridge(Z, yb - y0, best[0])
    vw = (V / (sd ** 2)).mean(axis=0)
    return {'mu': mu, 'sd': sd, 'b': b, 'y0': y0, 'vw': vw,
            'med': float(np.median(y)), 'lo': float(np.min(y)), 'hi': float(np.max(y)),
            'alpha': float(best[0]), 'lopo': lopo, 'lopo_best': round(float(best[1]), 4),
            'npat': n}


def pred_row(mdl, F):
    Z = (F - mdl['mu']) / mdl['sd']
    rel = 1.0 / (1.0 + mdl['vw'])
    return mdl['y0'] + (Z * rel) @ mdl['b']


def pred_clu(mdl, F, lab):
    Z = (F - mdl['mu']) / mdl['sd']
    out = np.full(Z.shape[0], mdl['y0'], dtype=float)
    for c in sorted(set(lab.tolist())):
        m = lab == c
        mm = max(int(m.sum()), 1)
        zc = Z[m].mean(axis=0)
        rel = 1.0 / (1.0 + mdl['vw'] / float(mm))
        out[m] = mdl['y0'] + float((zc * rel) @ mdl['b'])
    return out


def cluster(F, cn, k, seed):
    keep = [j for j, n in enumerate(cn) if 'test_time' not in str(n).lower()]
    A = F[:, keep].astype(float)
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    A = (A - mu) / sd
    k = max(1, min(int(k), A.shape[0]))
    km = KMeans(n_clusters=k, n_init=10, random_state=int(seed) % 100000)
    return np.asarray(km.fit_predict(A))


def gmae(pred, yt, gt):
    vals = []
    for k in sorted(set(gt)):
        m = np.array([s == k for s in gt])
        vals.append(float(np.mean(np.abs(pred[m] - yt[m]))))
    return float(np.mean(vals)) if vals else float('nan')


def fit_predict(train, inputs, seed):
    findings = {'design': 'patient_level_ridge_reliability_shrinkage_plus_transductive_cluster_smoothing',
                'fallback_used': False,
                'transductive_note': 'unlabeled evaluation X used only for KMeans grouping of rows; no evaluation labels used'}
    try:
        names = [str(n) for n in train['feature_names']]
        F, cn = design(train['X'], names)
        y = np.asarray(train['y'], dtype=float)
        g = [str(v) for v in train['groups']]
        en = [str(n) for n in inputs['feature_names']]
        Xi = np.asarray(inputs['X'], dtype=float)
        if en != names:
            idx = [en.index(n) for n in names]
            Xi = Xi[:, idx]
        G, _ = design(Xi, names)
        keys = sorted(set(g))
        nf = 4 if len(keys) >= 12 else 2
        folds = [[k for i, k in enumerate(keys) if i % nf == f] for f in range(nf)]
        modes = ['row', 'clu']
        cons = ['pm', 'med']
        ws = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        mse = {}
        gm = {}
        for f in range(nf):
            te = set(folds[f])
            mtr = np.array([s not in te for s in g])
            mte = ~mtr
            if int(mtr.sum()) < 20 or int(mte.sum()) < 20:
                continue
            gtr = [s for s, t in zip(g, mtr.tolist()) if t]
            gte = [s for s, t in zip(g, mte.tolist()) if t]
            mdl = fit_model(F[mtr], y[mtr], gtr)
            kte = max(1, min(12, int(round(float(mte.sum()) / 140.0))))
            lab = cluster(F[mte], cn, kte, seed)
            pr = {'row': pred_row(mdl, F[mte]), 'clu': pred_clu(mdl, F[mte], lab)}
            cv = {'pm': mdl['y0'], 'med': mdl['med']}
            yt = y[mte]
            for mo in modes:
                for c in cons:
                    for w in ws:
                        p = np.clip(w * pr[mo] + (1.0 - w) * cv[c], mdl['lo'], mdl['hi'])
                        key = '%s|%s|%.1f' % (mo, c, w)
                        mse.setdefault(key, []).append(float(np.mean((p - yt) ** 2)))
                        gm.setdefault(key, []).append(gmae(p, yt, gte))
        if not mse:
            raise RuntimeError('no usable internal folds')
        scored = []
        for key, v in mse.items():
            scored.append((float(np.sqrt(np.mean(v))), float(np.mean(gm[key])), key))
        scored.sort(key=lambda t: (t[0], t[2]))
        chosen = scored[0][2]
        mo, c, w = chosen.split('|')
        w = float(w)
        mdlf = fit_model(F, y, g)
        cvf = {'pm': mdlf['y0'], 'med': mdlf['med']}
        keval = max(1, min(12, int(round(float(G.shape[0]) / 140.0))))
        if w > 0.0 and mo == 'clu':
            labe = cluster(G, cn, keval, seed)
            base = pred_clu(mdlf, G, labe)
            csz = sorted([int((labe == u).sum()) for u in set(labe.tolist())])
        elif w > 0.0:
            base = pred_row(mdlf, G)
            csz = []
        else:
            base = np.full(G.shape[0], cvf[c], dtype=float)
            csz = []
        pred = np.clip(w * base + (1.0 - w) * cvf[c], mdlf['lo'], mdlf['hi'])
        pred = np.where(np.isfinite(pred), pred, cvf[c])
        rel = 1.0 / (1.0 + mdlf['vw'])
        order = np.argsort(-rel)
        findings['reliability_single_row_top'] = [[cn[int(j)], round(float(rel[int(j)]), 3)] for j in order[:8]]
        findings['reliability_single_row_bottom'] = [[cn[int(j)], round(float(rel[int(j)]), 3)] for j in order[-5:]]
        findings['n_train_patients'] = int(mdlf['npat'])
        findings['lopo_patient_mean_rmse'] = mdlf['lopo']
        findings['lopo_best_alpha'] = mdlf['alpha']
        findings['lopo_best_rmse'] = mdlf['lopo_best']
        findings['between_patient_signal_beats_constant'] = bool(mdlf['lopo_best'] < mdlf['lopo']['const'])
        findings['cv_top5_mode_const_weight_rmse_gmae'] = [[t[2], round(t[0], 4), round(t[1], 4)] for t in scored[:5]]
        findings['cv_constant_only'] = [[k2, round(float(np.sqrt(np.mean(mse[k2]))), 4)] for k2 in sorted(mse) if k2.endswith('|0.0')][:4]
        findings['chosen'] = {'mode': mo, 'constant': c, 'weight': w}
        findings['k_eval_clusters'] = int(keval)
        findings['eval_cluster_sizes'] = csz
        try:
            la = cluster(F, cn, min(28, len(keys)), seed)
            findings['train_cluster_ari_vs_patient'] = round(float(adjusted_rand_score(g, la.tolist())), 3)
        except Exception:
            findings['train_cluster_ari_vs_patient'] = None
        findings['pred_summary'] = {'min': round(float(pred.min()), 3), 'max': round(float(pred.max()), 3),
                                    'mean': round(float(pred.mean()), 3), 'sd': round(float(pred.std()), 3),
                                    'n': int(pred.shape[0])}
        out = [float(v) for v in pred.tolist()]
    except Exception as e:
        med = float(np.median(np.asarray(train['y'], dtype=float)))
        out = [med] * len(inputs['X'])
        findings = {'fallback_used': True, 'fallback_prediction': 'train_median_constant',
                    'error': str(e)[:300]}
    try:
        s = json.dumps(findings)
        if len(s.encode('utf-8')) > 8000:
            findings = {'fallback_used': findings.get('fallback_used', False),
                        'chosen': findings.get('chosen'),
                        'lopo_best_rmse': findings.get('lopo_best_rmse'),
                        'note': 'findings truncated for size'}
    except Exception:
        findings = {'fallback_used': True, 'error': 'findings_not_serializable'}
    return {'prediction': out, 'findings': findings}
