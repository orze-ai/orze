import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = np.asarray(X_train, dtype=float)
    Xe = np.asarray(X_eval, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    if Xe.ndim == 1:
        Xe = Xe.reshape(1, -1)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(1, -1)
    m = int(Xe.shape[0])
    n = int(y.shape[0])
    if m == 0:
        return np.zeros(0, dtype=float)
    if n == 0 or Xtr.shape[0] != n:
        return np.zeros(m, dtype=float)
    base = float(np.mean(y))
    if not np.isfinite(base):
        base = 0.0
    if base < 0.0:
        base = 0.0
    try:
        scaler = StandardScaler()
        Ztr = scaler.fit_transform(Xtr)
        Ze = scaler.transform(Xe)
        model = SVR(kernel="rbf", C=106.489793, gamma="scale", epsilon=0.1)
        model.fit(Ztr, y)
        p = np.asarray(model.predict(Ze), dtype=float).ravel()
    except Exception:
        p = np.full(m, base, dtype=float)
    if p.shape[0] != m or not np.all(np.isfinite(p)):
        p = np.full(m, base, dtype=float)
    center = float(np.mean(p))
    if not np.isfinite(center):
        center = base
    weight = 0.5
    out = center + weight * (p - center)
    out = np.where(np.isfinite(out), out, center)
    out = np.clip(out, 0.0, None)
    out = np.asarray(out, dtype=float).ravel()
    if out.shape[0] != m or not np.all(np.isfinite(out)):
        out = np.full(m, base, dtype=float)
    return out
