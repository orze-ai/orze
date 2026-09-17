import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


def _augment(X):
    X = np.asarray(X, dtype=float)
    eps = 1e-6
    cyl = np.maximum(X[:, 0], eps)
    disp = np.maximum(X[:, 1], eps)
    hp = np.maximum(X[:, 2], eps)
    wt = np.maximum(X[:, 3], eps)
    acc = X[:, 4]
    yr = X[:, 5]
    org = X[:, 6]
    cols = [
        np.log(disp),
        np.log(hp),
        np.log(wt),
        hp / wt,
        disp / cyl,
        hp / disp,
        acc,
        yr,
        cyl,
        (np.abs(org - 1.0) < 0.5).astype(float),
        (np.abs(org - 2.0) < 0.5).astype(float),
        (np.abs(org - 3.0) < 0.5).astype(float),
    ]
    return np.column_stack(cols)


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).ravel()
    Xev = np.asarray(X_eval, dtype=float)

    models = [
        TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(),
                                    SVR(C=10.0, gamma="scale", epsilon=0.1)),
            transformer=StandardScaler()),
        make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=2, include_bias=False),
                      StandardScaler(),
                      Ridge(alpha=1.0)),
        make_pipeline(StandardScaler(),
                      KNeighborsRegressor(n_neighbors=5, p=1,
                                          weights="uniform", n_jobs=1)),
        RandomForestRegressor(n_estimators=300, max_depth=8,
                              max_features=0.5, min_samples_leaf=1,
                              random_state=seed, n_jobs=1),
    ]

    log_member = make_pipeline(FunctionTransformer(_augment, validate=False),
                               StandardScaler(),
                               Ridge(alpha=1.0))
    if np.all(ytr > 0.0):
        models.append(TransformedTargetRegressor(regressor=log_member,
                                                 func=np.log,
                                                 inverse_func=np.exp,
                                                 check_inverse=False))
    else:
        models.append(log_member)

    preds = []
    for m in models:
        m.fit(Xtr, ytr)
        p = np.asarray(m.predict(Xev), dtype=float).ravel()
        preds.append(p)

    out = np.mean(np.vstack(preds), axis=0)
    fill = float(np.mean(ytr))
    out = np.where(np.isfinite(out), out, fill)
    lo = float(np.min(ytr)) - 5.0
    hi = float(np.max(ytr)) + 5.0
    return np.clip(out, lo, hi)
