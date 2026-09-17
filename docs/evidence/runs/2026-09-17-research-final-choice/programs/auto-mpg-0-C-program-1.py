import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


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
