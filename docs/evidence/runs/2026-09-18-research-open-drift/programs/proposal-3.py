import numpy as np
from sklearn.linear_model import LogisticRegression


def _bz(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


def _fit_lr(X, y, seed):
    m = LogisticRegression(C=1.0, max_iter=1500, class_weight="balanced",
                           solver="lbfgs", random_state=int(seed) % 100000)
    m.fit(X, y)
    return m


def fit_predict(train, X_eval, batch_eval, seed):
    Xtr = np.asarray(train["X"], dtype=float)
    ytr = np.asarray(train["y"], dtype=int)
    btr = np.asarray(train["batch"], dtype=int)
    Xev = np.asarray(X_eval, dtype=float)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xev = np.nan_to_num(Xev, nan=0.0, posinf=0.0, neginf=0.0)

    # per-batch z-scoring: each acquisition batch (train or eval) is scaled
    # using only its own rows (transductively legal, no labels used)
    Ztr = np.empty_like(Xtr)
    for b in np.unique(btr):
        m = btr == b
        Ztr[m] = _bz(Xtr[m]) if int(m.sum()) > 1 else Xtr[m]
    Zev = _bz(Xev) if Xev.shape[0] > 1 else Xev.copy()

    classes = np.unique(ytr)
    cidx = {int(c): i for i, c in enumerate(classes)}

    experts = []
    for b in np.unique(btr):
        m = btr == b
        if np.unique(ytr[m]).size >= 2 and int(m.sum()) >= 20:
            experts.append(_fit_lr(Ztr[m], ytr[m], seed))
    experts.append(_fit_lr(Ztr, ytr, seed))

    cover = np.zeros(classes.size)
    for mdl in experts:
        for c in mdl.classes_:
            cover[cidx[int(c)]] += 1.0
    cover = np.where(cover < 1.0, 1.0, cover)

    def scores(Z):
        S = np.zeros((Z.shape[0], classes.size))
        for mdl in experts:
            P = mdl.predict_proba(Z)
            for j, c in enumerate(mdl.classes_):
                S[:, cidx[int(c)]] += P[:, j]
        return S / cover[None, :]

    tmean = np.zeros((classes.size, Ztr.shape[1]))
    for c in classes:
        tmean[cidx[int(c)]] = Ztr[ytr == c].mean(axis=0)

    Zcur = Zev.copy()
    for _ in range(3):
        S = scores(Zcur)
        rs = S.sum(axis=1, keepdims=True)
        rs = np.where(rs < 1e-12, 1.0, rs)
        P = S / rs
        lab = np.argmax(P, axis=1)
        conf = P.max(axis=1)
        diffs = []
        for i in range(classes.size):
            m = (lab == i) & (conf >= 0.5)
            if int(m.sum()) >= 15:
                diffs.append(Zcur[m].mean(axis=0) - tmean[i])
        if len(diffs) < 2:
            break
        step = 0.7 * np.mean(np.asarray(diffs), axis=0)
        if (not np.all(np.isfinite(step))) or float(np.linalg.norm(step)) < 1e-4:
            break
        Zcur = Zcur - step

    pred = classes[np.argmax(scores(Zcur), axis=1)]
    return [int(v) for v in pred]
