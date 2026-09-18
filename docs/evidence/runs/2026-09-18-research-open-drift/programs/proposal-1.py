import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def _arr(d):
    X = np.asarray(d['X'], dtype=float)
    y = np.asarray(d['y'], dtype=int)
    b = np.asarray(d['batch'], dtype=int)
    c = np.asarray(d['concentration'], dtype=float)
    return X, y, b, c


def _rep(X, b, mode):
    Z = np.array(X, dtype=float)
    if 'log' in mode:
        Z = np.sign(Z) * np.log1p(np.abs(Z))
    if 'l2' in mode:
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        n[n < 1e-12] = 1.0
        Z = Z / n
    if 'bz' in mode:
        for bb in np.unique(b):
            m = b == bb
            mu = Z[m].mean(axis=0)
            sd = Z[m].std(axis=0)
            sd[sd < 1e-9] = 1.0
            Z[m] = (Z[m] - mu) / sd
    return Z


def _bal(yt, yp):
    ks = sorted(set(yt.tolist()))
    rec = []
    for k in ks:
        m = yt == k
        rec.append(float((yp[m] == k).mean()))
    return float(1.0 - float(np.mean(rec))), dict((str(k), round(r, 3)) for k, r in zip(ks, rec))


def _fit(Xtr, ytr, seed, it):
    sc = StandardScaler()
    A = sc.fit_transform(Xtr)
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=it, random_state=seed)
    clf.fit(A, ytr)
    return sc, clf


def analyze(data, history, seed):
    out = {}
    out['claim_status'] = 'diagnostic tables only; causal wording is hypothesis until supported here'
    Xt, yt, bt, ct = _arr(data['train'])
    Xd, yd, bd, cd = _arr(data['development'])
    X = np.vstack([Xt, Xd])
    y = np.concatenate([yt, yd])
    b = np.concatenate([bt, bd])
    c = np.concatenate([ct, cd])
    batches = sorted(set(b.tolist()))
    out['shapes'] = {'train_rows': int(Xt.shape[0]), 'dev_rows': int(Xd.shape[0]), 'features': int(X.shape[1]), 'batches': batches}
    cnt = {}
    conc = {}
    for bb in batches:
        m = b == bb
        cnt[str(bb)] = [int(((y == k) & m).sum()) for k in range(1, 7)]
        row = []
        for k in range(1, 7):
            mk = m & (y == k)
            if int(mk.sum()) == 0:
                row.append(None)
            else:
                row.append([round(float(np.percentile(c[mk], 10)), 1), round(float(np.percentile(c[mk], 90)), 1)])
        conc[str(bb)] = row
    out['class_counts_by_batch_c1..c6'] = cnt
    out['conc_p10_p90_by_class_batch'] = conc
    idx0 = np.arange(0, 128, 8)
    scale_all = {}
    scale_c1 = {}
    for bb in batches:
        m = b == bb
        scale_all[str(bb)] = [round(float(np.median(np.abs(X[m][:, j]))), 3) for j in idx0[:8]]
        m1 = m & (y == 1)
        if int(m1.sum()) >= 5:
            scale_c1[str(bb)] = [round(float(np.median(np.abs(X[m1][:, j]))), 3) for j in idx0[:8]]
        else:
            scale_c1[str(bb)] = None
    out['median_abs_blockf0_sensors1_8_allclasses'] = scale_all
    out['median_abs_blockf0_sensors1_8_class1only'] = scale_c1
    modes = ['raw', 'bz', 'log', 'log_bz', 'log_l2_bz']
    tr_mask = np.isin(b, [1, 2, 3, 4])
    transfer = {}
    c6_dest = {}
    for mode in modes:
        Z = _rep(X, b, mode)
        rec = {}
        sc, clf = _fit(Z[tr_mask], y[tr_mask], seed, 900)
        for bb in [5, 6, 7]:
            m = b == bb
            if int(m.sum()) == 0:
                continue
            yp = clf.predict(sc.transform(Z[m]))
            be, rr = _bal(y[m], yp)
            rec['dev%d_bal_err' % bb] = round(be, 4)
            rec['dev%d_recall' % bb] = rr
            m6 = m & (y == 6)
            if int(m6.sum()) > 0:
                p6 = clf.predict(sc.transform(Z[m6]))
                c6_dest.setdefault(mode, {})['dev%d' % bb] = dict((str(k), int((p6 == k).sum())) for k in range(1, 7) if int((p6 == k).sum()) > 0)
        lobo = []
        for hb in [1, 2, 3, 4]:
            trm = np.isin(b, [x for x in [1, 2, 3, 4] if x != hb])
            tem = b == hb
            if int(tem.sum()) == 0 or len(set(y[trm].tolist())) < 2:
                continue
            s2, c2 = _fit(Z[trm], y[trm], seed, 500)
            be, _ = _bal(y[tem], c2.predict(s2.transform(Z[tem])))
            lobo.append(be)
        if lobo:
            rec['train_lobo_mean_bal_err'] = round(float(np.mean(lobo)), 4)
            rec['train_lobo_each'] = [round(v, 3) for v in lobo]
        mean_dev = [rec[k] for k in rec if k.endswith('_bal_err') and k.startswith('dev')]
        if mean_dev:
            rec['dev_mean_bal_err'] = round(float(np.mean(mean_dev)), 4)
        transfer[mode] = rec
    out['transfer_logreg_by_representation'] = transfer
    out['class6_prediction_destinations'] = c6_dest
    out['per_batch_stats_are_transductive'] = 'bz modes use only that batch own mean/std, so they are reproducible by a method given one unlabeled eval batch'
    hist = []
    try:
        for h in (history or []):
            hist.append({'task_id': h.get('task_id'), 'valid': h.get('valid')})
    except Exception:
        hist = []
    out['history_seen'] = hist[-6:]
    return out
