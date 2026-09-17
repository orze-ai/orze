import numpy as np


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    """Predict the training-mean constant for every evaluation row.

    Under independently permuted labels there is no row-specific signal to
    exploit, so the MSE-optimal row-independent mapping is the constant equal
    to the conditional mean, estimated only from training rows.
    """
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    if Xe.ndim == 1:
        n_eval = 1 if Xe.size else 0
    else:
        n_eval = Xe.shape[0]
    finite = y[np.isfinite(y)] if y.size else y
    if finite.size == 0:
        level = 0.0
    else:
        level = float(np.mean(finite))
    if not np.isfinite(level):
        level = 0.0
    return np.full(int(n_eval), level, dtype=float)
