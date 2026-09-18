import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def _tf(X):
    return np.sign(X) * np.log1p(np.abs(X))


def _rstd(X):
    med = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    s = q3 - q1
    s = np.where(s < 1e-8, 1.0, s)
    return med, s


def _wh(Z, shrink=0.3):
    C = np.cov(Z, rowvar=False)
    d = C.shape[0]
    C = (1.0 - shrink) * C + shrink * (np.trace(C) / max(d, 1)) * np.eye(d)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-8)
    return (V * (w ** -0.5)) @ V.T


def _probs(A, y, B):
    cls = np.unique(y)
    pri = np.ones(len(cls), dtype=float) / float(len(cls))
    out = []
    try:
        lr = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced')
        lr.fit(A, y)
        out.append(lr.predict_proba(B))
    except Exception:
        pass
    try:
        ld = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=pri)
        ld.fit(A, y)
        out.append(ld.predict_proba(B))
    except Exception:
        pass
    return out


def fit_predict(train, X_eval, batch_eval, seed=0):
    Xtr = np.asarray(train['X'], dtype=float)
    ytr = np.asarray(train['y'], dtype=int)
    btr = np.asarray(train['batch'], dtype=int)
    Xev = np.asarray(X_eval, dtype=float)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xev = np.nan_to_num(Xev, nan=0.0, posinf=0.0, neginf=0.0)
    cls = np.unique(ytr)
    try:
        Ltr = _tf(Xtr)
        Lev = _tf(Xev)
        gmed, gs = _rstd(Ltr)
        Ztr = np.empty_like(Ltr)
        for b in np.unique(btr):
            m = (btr == b)
            if int(m.sum()) < 20:
                med, s = gmed, gs
            else:
                med, s = _rstd(Ltr[m])
            Ztr[m] = (Ltr[m] - med) / s
        if Lev.shape[0] < 20:
            med, s = gmed, gs
        else:
            med, s = _rstd(Lev)
        Zev = (Lev - med) / s
        Ztr = np.clip(Ztr, -8.0, 8.0)
        Zev = np.clip(Zev, -8.0, 8.0)
        Atr = Ztr @ _wh(Ztr)
        Aev = Zev @ _wh(Zev)
        acc = []
        for S, T in ((Ztr, Zev), (Atr, Aev)):
            for p in _probs(S, ytr, T):
                acc.append(p)
        if len(acc) == 0:
            raise RuntimeError('no model fitted')
        P = np.mean(np.stack(acc, axis=0), axis=0)
        idx = np.argmax(P, axis=1)
        return [int(cls[i]) for i in idx]
    except Exception:
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        lr = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced')
        lr.fit((Xtr - mu) / sd, ytr)
        return [int(v) for v in lr.predict((Xev - mu) / sd)]
