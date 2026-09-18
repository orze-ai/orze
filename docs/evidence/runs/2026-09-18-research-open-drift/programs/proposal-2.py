import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


def _log(Z):
    return np.sign(Z) * np.log1p(np.abs(Z))


def _rz(Z):
    med = np.median(Z, axis=0)
    iqr = np.percentile(Z, 75, axis=0) - np.percentile(Z, 25, axis=0)
    iqr = np.where(iqr < 1e-9, 1.0, iqr)
    return (Z - med) / iqr


def _bal(yt, yp):
    rec = {}
    errs = []
    for k in np.unique(yt):
        m = yt == k
        r = float(np.mean(yp[m] == k))
        rec[int(k)] = round(r, 3)
        errs.append(1.0 - r)
    return round(float(np.mean(errs)), 4), rec


def _fit(Xtr, ytr, it=300):
    return LogisticRegression(C=1.0, max_iter=it, class_weight='balanced').fit(Xtr, ytr)


def _sub(X, y, cap, rng):
    if X.shape[0] <= cap:
        return X, y
    idx = rng.choice(X.shape[0], cap, replace=False)
    return X[idx], y[idx]


def analyze(data, history, seed):
    rng = np.random.RandomState(seed)
    Xs = []
    ys = []
    bs = []
    for part in ('train', 'development'):
        d = data[part]
        Xs.append(np.asarray(d['X'], dtype=float))
        ys.append(np.asarray(d['y'], dtype=int))
        bs.append(np.asarray(d['batch'], dtype=int))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    b = np.concatenate(bs)
    L = _log(X)
    batches = sorted(set(b.tolist()))
    Z = {}
    Y = {}
    for bb in batches:
        m = b == bb
        Z[bb] = _rz(L[m])
        Y[bb] = y[m]
    out = {}
    out['claim_status'] = 'diagnostic measurements only; mechanism wording stays hypothesis'
    out['rep'] = 'signed log1p then per-batch median/IQR scaling (uses only that batch, transductively legal)'
    out['oracle_note'] = 'common/class shift oracles use dev labels as upper bounds; a frozen method may not use them'
    hs = history or []
    out['history_seen'] = [{'task_id': h.get('task_id'), 'valid': h.get('valid')} for h in hs][-6:]
    pw = {}
    c6 = {}
    for a in batches:
        cnt = {int(k): int(np.sum(Y[a] == k)) for k in set(Y[a].tolist())}
        ok = np.array([cnt[int(v)] >= 5 for v in Y[a]])
        Xa, ya = _sub(Z[a][ok], Y[a][ok], 900, rng)
        if len(set(ya.tolist())) < 2:
            continue
        cl = _fit(Xa, ya, 200)
        row = {}
        for t in batches:
            if t == a:
                continue
            yp = cl.predict(Z[t])
            sh = np.array([cnt.get(int(v), 0) >= 5 for v in Y[t]])
            if not sh.any():
                continue
            be, rec = _bal(Y[t][sh], yp[sh])
            row[str(t)] = be
            if 6 in rec:
                c6[str(a) + '>' + str(t)] = rec[6]
        pw[str(a)] = row
    out['pairwise_bal_err_row_train_col_test_sharedclasses'] = pw
    out['class6_recall_pairwise'] = c6
    tr = [x for x in batches if x <= 4]
    dv = [x for x in batches if x >= 5]
    Xtr = np.vstack([Z[x] for x in tr])
    ytr = np.concatenate([Y[x] for x in tr])
    clf = _fit(Xtr, ytr, 400)
    ctr = {}
    for k in np.unique(ytr):
        if np.sum(ytr == k) >= 5:
            ctr[int(k)] = Xtr[ytr == k].mean(axis=0)
    dec = {}
    orc = {}
    for t in dv:
        Zt = Z[t]
        yt = Y[t]
        cd = {}
        for k in np.unique(yt):
            if np.sum(yt == k) >= 5:
                cd[int(k)] = Zt[yt == k].mean(axis=0)
        S = [k for k in sorted(cd) if k in ctr]
        if len(S) < 2:
            continue
        D = np.array([cd[k] - ctr[k] for k in S])
        mshift = D.mean(axis=0)
        R = D - mshift
        dec[str(t)] = {'shared': S,
                       'common_norm': round(float(np.linalg.norm(mshift)), 3),
                       'mean_resid_norm': round(float(np.mean(np.linalg.norm(R, axis=1))), 3),
                       'resid_by_class': {str(S[i]): round(float(np.linalg.norm(R[i])), 3) for i in range(len(S))}}
        be0, r0 = _bal(yt, clf.predict(Zt))
        be1, r1 = _bal(yt, clf.predict(Zt - mshift))
        Zc = Zt.copy()
        for i, k in enumerate(S):
            sel = yt == k
            Zc[sel] = Zc[sel] - D[i]
        be2, r2 = _bal(yt, clf.predict(Zc))
        orc[str(t)] = {'none': [be0, r0], 'common_oracle': [be1, r1], 'classwise_oracle': [be2, r2]}
    out['centroid_drift_decomposition'] = dec
    out['alignment_oracles'] = orc
    km = {}
    for t in dv:
        Zt = Z[t]
        yt = Y[t]
        ks = np.unique(yt)
        n = len(ks)
        lab = KMeans(n_clusters=n, n_init=5, random_state=seed).fit_predict(Zt)
        M = np.zeros((n, n))
        for i, k in enumerate(ks):
            tot = max(1, int(np.sum(yt == k)))
            for j in range(n):
                M[i, j] = float(np.sum((yt == k) & (lab == j))) / tot
        ri, ci = linear_sum_assignment(-M)
        mp = {int(ci[i]): int(ks[ri[i]]) for i in range(len(ri))}
        yp = np.array([mp[int(v)] for v in lab])
        be, rec = _bal(yt, yp)
        km[str(t)] = {'hungarian_bal_err': be, 'recall': rec}
    out['unsup_kmeans_structure_upper_bound'] = km
    means = {}
    for x in tr:
        for k in np.unique(Y[x]):
            if np.sum(Y[x] == k) >= 5:
                means.setdefault(int(k), []).append(Z[x][Y[x] == k].mean(axis=0))
    wb = []
    cm = []
    for k in sorted(means):
        A = np.array(means[k])
        if A.shape[0] >= 2:
            wb.append(A.var(axis=0))
        cm.append(A.mean(axis=0))
    wbv = np.mean(np.array(wb), axis=0) if len(wb) else np.ones(Xtr.shape[1])
    bcv = np.array(cm).var(axis=0)
    score = bcv / (wbv + 1e-6)
    order = np.argsort(-score)
    st = {'all128': {str(t): _bal(Y[t], clf.predict(Z[t]))[0] for t in dv}}
    for kk in (24, 48, 96):
        sel = order[:kk]
        c2 = _fit(Xtr[:, sel], ytr, 400)
        res = {}
        for t in dv:
            be, rec = _bal(Y[t], c2.predict(Z[t][:, sel]))
            res[str(t)] = [be, rec.get(6)]
        st['top' + str(kk)] = res
    st['top24_idx'] = [int(v) for v in order[:24]]
    st['stability_score_def'] = 'between-class var of class means / mean across-batch var of per-batch class means, train 1..4 only'
    out['batch_stable_feature_selection'] = st
    return out
