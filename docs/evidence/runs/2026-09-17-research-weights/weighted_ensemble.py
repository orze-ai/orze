import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV

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


def _fold_masks(n, groups, k, rng):
    if groups is not None:
        g = np.asarray(groups).ravel()
        u = np.unique(g)
        if len(u) >= k:
            order = rng.permutation(len(u))
            masks = []
            for chunk in np.array_split(order, k):
                masks.append(np.isin(g, u[chunk]))
            return masks
    idx = rng.permutation(n)
    base = np.arange(n)
    masks = []
    for chunk in np.array_split(idx, k):
        masks.append(np.isin(base, chunk))
    return masks


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    Xa = _aug(X)
    Xea = _aug(Xe)
    n = X.shape[0]
    m = 6
    reps = 2
    s = np.zeros((n, m), dtype=float)
    c = np.zeros(n, dtype=float)
    for r in range(reps):
        rng = np.random.default_rng(int(seed) + 977 * r)
        for mask in _fold_masks(n, train_groups, 5, rng):
            va = np.where(mask)[0]
            tr = np.where(~mask)[0]
            if len(tr) < 20 or len(va) == 0:
                continue
            fold = _models(seed)
            for j in range(m):
                kind, mdl = fold[j]
                A = Xa if kind == "aug" else X
                mdl.fit(A[tr], y[tr])
                s[va, j] += np.asarray(mdl.predict(A[va]), dtype=float).ravel()
            c[va] += 1.0
    w = None
    ok = c > 0
    if int(ok.sum()) >= 6 * m:
        A = s[ok] / c[ok][:, None]
        b = y[ok]
        if np.all(np.isfinite(A)) and np.all(np.isfinite(b)):
            mse = np.mean((A - b[:, None]) ** 2, axis=0)
            inv = 1.0 / np.maximum(mse, 1e-6) ** 2
            w_inv = inv / float(inv.sum())
            blend = LinearRegression(positive=True, fit_intercept=False)
            blend.fit(A, b)
            cand = np.asarray(blend.coef_, dtype=float).ravel()
            t = float(cand.sum())
            if np.all(np.isfinite(cand)) and t > 1e-8:
                w = 0.75 * (cand / t) + 0.25 * w_inv
            else:
                w = w_inv
    if w is None:
        w = np.ones(m, dtype=float) / float(m)
    preds = np.zeros((Xe.shape[0], m), dtype=float)
    full = _models(seed)
    for j in range(m):
        kind, mdl = full[j]
        A = Xa if kind == "aug" else X
        B = Xea if kind == "aug" else Xe
        mdl.fit(A, y)
        preds[:, j] = np.asarray(mdl.predict(B), dtype=float).ravel()
    out = np.asarray(preds.dot(w), dtype=float).ravel()
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, float(np.mean(y)))
    return out
