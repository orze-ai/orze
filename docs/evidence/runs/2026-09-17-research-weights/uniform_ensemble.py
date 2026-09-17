import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV

LOGCOLS = [0, 1, 2, 6, 7]


def _aug(X):
    X = np.asarray(X, dtype=float)
    Z = X[:, LOGCOLS]
    Z = np.where(Z > 0.0, Z, 0.0)
    return np.hstack([X, np.log1p(Z)])


def _models(seed):
    return [
        ("raw", RandomForestRegressor(n_estimators=300, max_features=0.5,
                                      min_samples_leaf=2, random_state=seed, n_jobs=1)),
        ("raw", ExtraTreesRegressor(n_estimators=300, max_features=0.5,
                                    min_samples_leaf=1, random_state=seed, n_jobs=1)),
        ("raw", HistGradientBoostingRegressor(learning_rate=0.03, max_leaf_nodes=7,
                                              min_samples_leaf=40, l2_regularization=0.0,
                                              max_iter=300, early_stopping=False,
                                              random_state=seed)),
        ("raw", make_pipeline(StandardScaler(), SVR(C=4.5, gamma="auto", epsilon=0.1))),
        ("raw", make_pipeline(StandardScaler(),
                              KNeighborsRegressor(n_neighbors=9, p=1, weights="distance", n_jobs=1))),
        ("aug", make_pipeline(StandardScaler(),
                              PolynomialFeatures(degree=2, include_bias=False),
                              StandardScaler(),
                              RidgeCV(alphas=np.logspace(-1.0, 3.0, 13)))),
    ]


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    Xa, Xea = _aug(X), _aug(Xe)
    predictions = []
    for kind, model in _models(seed):
        train = Xa if kind == "aug" else X
        evaluation = Xea if kind == "aug" else Xe
        model.fit(train, y)
        predictions.append(np.asarray(model.predict(evaluation), dtype=float).ravel())
    out = np.mean(np.column_stack(predictions), axis=1)
    return np.where(np.isfinite(out), out, float(np.mean(y))).astype(float).ravel()
