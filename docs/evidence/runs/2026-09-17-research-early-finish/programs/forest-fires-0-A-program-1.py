import numpy as np
from sklearn.svm import SVR


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).ravel()
    Xev = np.asarray(X_eval, dtype=float)
    if Xev.ndim == 1:
        Xev = Xev.reshape(1, -1)
    n_eval = Xev.shape[0]
    fallback = float(np.mean(ytr)) if ytr.size else 0.0
    if not np.isfinite(fallback):
        fallback = 0.0
    members = [
        (13.889906, 0.635832, 0.35),
        (12.465816, 0.8768089, 0.35),
        (8.125237, "auto", 0.15),
        (28.336489, "auto", 0.15),
    ]
    acc = np.zeros(n_eval, dtype=float)
    total = 0.0
    for C, gamma, w in members:
        try:
            model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=0.1, tol=1e-3, cache_size=200, max_iter=-1)
            model.fit(Xtr, ytr)
            p = np.asarray(model.predict(Xev), dtype=float).ravel()
        except Exception:
            continue
        if p.shape[0] != n_eval:
            continue
        p = np.where(np.isfinite(p), p, fallback)
        acc = acc + w * p
        total += w
    if total <= 0.0:
        out = np.full(n_eval, fallback, dtype=float)
    else:
        out = acc / total
    out = np.where(np.isfinite(out), out, fallback)
    out = np.clip(out, 0.0, None)
    return np.asarray(out, dtype=float).ravel()
