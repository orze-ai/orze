import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

SPECS = (
    (13.889906, 0.635832, False),
    (12.465816, 0.8768089, False),
    (8.125237, 'auto', False),
    (8.585495, 'scale', True),
)


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    n_eval = Xe.shape[0]
    mu = float(np.mean(y)) if y.size > 0 else 0.0
    base = max(mu, 0.0)
    preds = []
    for C, g, scale in SPECS:
        try:
            if scale:
                sc = StandardScaler().fit(X)
                Xt = sc.transform(X)
                Xv = sc.transform(Xe)
            else:
                Xt = X
                Xv = Xe
            model = SVR(kernel='rbf', C=float(C), gamma=g, epsilon=0.1)
            model.fit(Xt, y)
            p = np.asarray(model.predict(Xv), dtype=float).ravel()
            if p.shape[0] == n_eval and np.all(np.isfinite(p)):
                preds.append(p)
        except Exception:
            continue
    if len(preds) == 0:
        out = np.full(n_eval, base, dtype=float)
    else:
        out = np.mean(np.vstack(preds), axis=0)
    out = np.asarray(out, dtype=float).ravel()
    out = np.maximum(out, 0.0)
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = base
    return out
