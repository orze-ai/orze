import numpy as np


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    """Constant predictor: training-sample mean for every evaluation row.

    Rationale: under the disclosed within-split label permutation there is no
    row-level signal to learn, and the squared-error optimal row-independent
    mapping is the marginal mean of the training targets. Everything is fit on
    training rows only; X_eval is used solely for its row count.
    """
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    if Xe.ndim == 1:
        n_eval = 1 if Xe.size else 0
    else:
        n_eval = int(Xe.shape[0])
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        c = 0.0
    else:
        c = float(np.mean(finite))
    if not np.isfinite(c):
        c = 0.0
    return np.full(n_eval, c, dtype=float)
