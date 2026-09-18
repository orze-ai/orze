import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


def _clean(X):
    X = np.asarray(X, dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _batch_z(X, b):
    Z = np.empty_like(X)
    b = np.asarray(b)
    for u in np.unique(b):
        m = (b == u)
        blk = X[m]
        mu = blk.mean(axis=0)
        sd = blk.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Z[m] = (blk - mu) / sd
    return Z


def fit_predict(train, X_eval, batch_eval, seed=0):
    Xtr = _clean(train['X'])
    ytr = np.asarray(train['y'], dtype=int)
    btr = np.asarray(train['batch'], dtype=int)
    Xev = _clean(X_eval)
    bev = np.asarray(batch_eval)
    if Xev.ndim == 1:
        Xev = Xev.reshape(1, -1)
    Ztr = _batch_z(Xtr, btr)
    Zev = _batch_z(Xev, bev)
    k = int(min(40, Ztr.shape[1], max(2, Ztr.shape[0] - 1)))
    pca = PCA(n_components=k, random_state=int(seed))
    pca.fit(np.vstack([Ztr, Zev]))
    Ptr = pca.transform(Ztr)
    Pev = pca.transform(Zev)
    sc = Ptr.std(axis=0)
    sc = np.where(sc < 1e-8, 1.0, sc)
    Ptr = Ptr / sc
    Pev = Pev / sc
    classes = np.unique(ytr)
    C = int(classes.shape[0])
    cidx = {}
    for i in range(C):
        cidx[int(classes[i])] = i
    probs = np.zeros((Pev.shape[0], C))
    counts = np.zeros(C)
    groups = [(-1, np.ones(Ptr.shape[0], dtype=bool))]
    for u in np.unique(btr):
        groups.append((int(u), btr == u))
    for name, m in groups:
        yy = ytr[m]
        if yy.shape[0] < 10 or np.unique(yy).shape[0] < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
        clf.fit(Ptr[m], yy)
        pp = clf.predict_proba(Pev)
        w = 2.0 if name == -1 else 1.0
        for j in range(clf.classes_.shape[0]):
            i = cidx[int(clf.classes_[j])]
            probs[:, i] += w * pp[:, j]
            counts[i] += w
    counts = np.where(counts < 1e-9, 1.0, counts)
    ens = probs / counts[None, :]
    ssum = ens.sum(axis=1, keepdims=True)
    ens = ens / np.maximum(ssum, 1e-12)
    log_ens = np.log(np.clip(ens, 1e-9, None))
    mu_tr = np.zeros((C, k))
    for c in classes:
        mu_tr[cidx[int(c)]] = Ptr[ytr == c].mean(axis=0)
    W = np.zeros((k, k))
    n = 0
    for c in classes:
        A = Ptr[ytr == c]
        d = A - A.mean(axis=0)
        W += d.T @ d
        n += int(A.shape[0])
    W = W / max(n - C, 1)
    W = W + (np.trace(W) / k * 0.05 + 1e-6) * np.eye(k)
    try:
        Winv = np.linalg.inv(W)
    except Exception:
        Winv = np.linalg.pinv(W)

    def logl(Q, mu):
        out = np.empty((Q.shape[0], mu.shape[0]))
        for i in range(mu.shape[0]):
            d = Q - mu[i]
            out[:, i] = -0.5 * np.sum((d @ Winv) * d, axis=1)
        return out

    Q = Pev.copy()
    post = ens.copy()
    for _ in range(3):
        hard = np.argmax(post, axis=1)
        conf = np.max(post, axis=1)
        ds = []
        for i in range(C):
            m = (hard == i) & (conf >= 0.5)
            if int(m.sum()) >= 5:
                ds.append(Q[m].mean(axis=0) - mu_tr[i])
        if len(ds) == 0:
            break
        dv = np.mean(np.asarray(ds), axis=0)
        Q = Q - 0.7 * dv
        Lt = logl(Q, mu_tr)
        Lt = Lt - Lt.max(axis=1, keepdims=True)
        post = np.exp(Lt)
        post = post / np.maximum(post.sum(axis=1, keepdims=True), 1e-12)
    L0 = logl(Q, mu_tr)
    if C >= 2:
        srt = np.sort(L0, axis=1)
        gap = float(np.median(srt[:, -1] - srt[:, -2]))
    else:
        gap = 1.0
    beta = 3.0 / max(gap, 1e-6)
    beta = float(np.clip(beta, 1e-4, 1.0))
    mu = mu_tr.copy()
    tau = 15.0
    for _ in range(12):
        L = beta * logl(Q, mu) + log_ens
        L = L - L.max(axis=1, keepdims=True)
        R = np.exp(L)
        R = R / np.maximum(R.sum(axis=1, keepdims=True), 1e-12)
        Nc = R.sum(axis=0)
        newmu = (R.T @ Q + tau * mu_tr) / (Nc + tau)[:, None]
        step = float(np.max(np.abs(newmu - mu)))
        mu = newmu
        if step < 1e-4:
            break
    score = beta * logl(Q, mu) + log_ens
    pred = classes[np.argmax(score, axis=1)]
    return [int(v) for v in pred]
