import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def _zb(X, b):
    Z = np.array(X, dtype=np.float64, copy=True)
    for u in np.unique(b):
        m = (b == u)
        sub = Z[m]
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Z[m] = (sub - mu) / sd
    return Z


def _experts(X, y, b):
    out = []
    m0 = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    m0.fit(X, y)
    out.append(m0)
    for u in np.unique(b):
        m = (b == u)
        if int(m.sum()) < 40:
            continue
        if np.unique(y[m]).size < 2:
            continue
        e = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        e.fit(X[m], y[m])
        out.append(e)
    return out


def _eproba(experts, X):
    acc = np.zeros((X.shape[0], 6))
    cnt = np.zeros(6)
    for e in experts:
        p = e.predict_proba(X)
        for j, c in enumerate(e.classes_):
            k = int(c) - 1
            acc[:, k] += p[:, j]
            cnt[k] += 1.0
    cnt = np.where(cnt <= 0, 1.0, cnt)
    P = np.clip(acc / cnt, 1e-12, None)
    return P / P.sum(axis=1, keepdims=True)


def fit_predict(train, X_eval, batch_eval, seed):
    Xtr = np.asarray(train["X"], dtype=np.float64)
    ytr = np.asarray(train["y"], dtype=int)
    btr = np.asarray(train["batch"], dtype=int)
    Xev = np.asarray(X_eval, dtype=np.float64)
    bev = np.asarray(batch_eval).ravel()
    if bev.shape[0] != Xev.shape[0]:
        bev = np.zeros(Xev.shape[0], dtype=int)
    Ztr = _zb(Xtr, btr)
    Zev = _zb(Xev, bev)
    nc = int(min(40, Ztr.shape[1], Ztr.shape[0] + Zev.shape[0] - 1))
    pca = PCA(n_components=nc, whiten=True, random_state=0)
    pca.fit(np.vstack([Ztr, Zev]))
    Ttr = pca.transform(Ztr)
    Tev = pca.transform(Zev)
    experts = _experts(Ttr, ytr, btr)
    Pb = _eproba(experts, Tev)
    base = np.argmax(Pb, axis=1) + 1
    mu_t = {}
    for c in range(1, 7):
        rows = []
        for u in np.unique(btr):
            m = (btr == u) & (ytr == c)
            if int(m.sum()) >= 3:
                rows.append(Ttr[m].mean(axis=0))
        if rows:
            mu_t[c] = np.mean(np.vstack(rows), axis=0)
    conf = Pb.max(axis=1)
    ref = []
    mu_e = {}
    for c in range(1, 7):
        if c not in mu_t:
            continue
        m = (base == c) & (conf >= 0.90)
        if int(m.sum()) >= 15:
            mu_e[c] = Tev[m].mean(axis=0)
            ref.append(c)
    if len(ref) < 3:
        return [int(v) for v in base]
    Mt = np.vstack([mu_t[c] for c in ref])
    D = np.vstack([mu_e[c] - mu_t[c] for c in ref])
    mbar = Mt.mean(axis=0)
    A = Mt - mbar
    b0 = D.mean(axis=0)
    Dc = D - b0
    d = A.shape[1]
    lam = 3.0 * max(1e-6, float(np.trace(A.T.dot(A))) / float(d))
    W = np.linalg.solve(A.T.dot(A) + lam * np.eye(d), A.T.dot(Dc))
    Xa = [Ttr]
    ya = [ytr]
    for c in range(1, 7):
        if c not in mu_t:
            continue
        dc = b0 + (mu_t[c] - mbar).dot(W)
        m = (ytr == c)
        if int(m.sum()) == 0:
            continue
        Xa.append(Ttr[m] + dc)
        ya.append(ytr[m])
    Xa = np.vstack(Xa)
    ya = np.concatenate(ya)
    ad = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    ad.fit(Xa, ya)
    Pa = np.zeros((Tev.shape[0], 6))
    pp = ad.predict_proba(Tev)
    for j, c in enumerate(ad.classes_):
        Pa[:, int(c) - 1] = pp[:, j]
    Pa = np.clip(Pa, 1e-12, None)
    Pa = Pa / Pa.sum(axis=1, keepdims=True)
    L = 0.5 * np.log(Pb) + 0.5 * np.log(Pa)
    return [int(v) for v in (np.argmax(L, axis=1) + 1)]
