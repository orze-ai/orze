import math
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _specs():
    out = []
    for C in (3.0, 10.0, 30.0, 100.0):
        for g in ("scale", 0.1, 0.2, 0.4):
            out.append(("svm", {"C": C, "gamma": g}))
    for leaf in (1, 2):
        for mf in (0.6, 1.0):
            out.append(("trees", {"leaf": leaf, "mf": mf}))
    for lr in (0.05, 0.08):
        out.append(("boost", {"lr": lr}))
    for k in (4, 8):
        out.append(("knn", {"k": k}))
    out.append(("ridge", {"alpha": 1.0}))
    return out


def _build(kind, p, seed):
    if kind == "svm":
        return Pipeline([("s", StandardScaler()),
                         ("m", SVR(kernel="rbf", C=p["C"], gamma=p["gamma"], epsilon=0.1))])
    if kind == "trees":
        return ExtraTreesRegressor(n_estimators=200, min_samples_leaf=p["leaf"],
                                   max_features=p["mf"], random_state=seed, n_jobs=1)
    if kind == "boost":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=p["lr"],
                                             max_leaf_nodes=31, min_samples_leaf=20,
                                             l2_regularization=1.0, early_stopping=False,
                                             random_state=seed)
    if kind == "knn":
        return Pipeline([("s", StandardScaler()),
                         ("m", KNeighborsRegressor(n_neighbors=p["k"], weights="distance", p=2))])
    return Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=p["alpha"]))])


def fit_predict(X_train, y_train, X_eval, seed, train_groups=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    Xe = np.asarray(X_eval, dtype=float)
    n = X.shape[0]
    base = float(np.mean(y)) if n > 0 else 0.0
    fallback = np.full(Xe.shape[0], base)
    if n < 20:
        return fallback
    if train_groups is None:
        g = np.arange(n)
    else:
        g = np.unique(np.asarray(train_groups), return_inverse=True)[1]
    k = 5
    if len(np.unique(g)) >= k:
        folds = list(GroupKFold(n_splits=k).split(X, y, g))
    else:
        folds = list(KFold(n_splits=k, shuffle=True, random_state=1729).split(X))
    specs = _specs()
    oof = np.zeros((len(specs), n))
    score = np.full(len(specs), np.inf)
    for i in range(len(specs)):
        kind, p = specs[i]
        good = True
        for tr, va in folds:
            try:
                mdl = _build(kind, p, seed)
                mdl.fit(X[tr], y[tr])
                pr = np.asarray(mdl.predict(X[va]), dtype=float)
            except Exception:
                good = False
                break
            if pr.shape[0] != va.shape[0] or not np.all(np.isfinite(pr)):
                good = False
                break
            oof[i, va] = pr
        if good:
            score[i] = float(np.mean((oof[i] - y) ** 2))
    best = {}
    for i in range(len(specs)):
        if not np.isfinite(score[i]):
            continue
        kind = specs[i][0]
        if kind not in best or score[i] < score[best[kind]]:
            best[kind] = i
    cand = sorted(best.values(), key=lambda j: score[j])[:4]
    if len(cand) == 0:
        return fallback
    chosen = [cand[0]]
    bs = float(score[cand[0]])
    m = len(cand)
    for mask in range(1, 1 << m):
        sel = [cand[j] for j in range(m) if (mask >> j) & 1]
        pred = np.mean(oof[sel], axis=0)
        s = float(np.mean((pred - y) ** 2))
        if s < bs - 1e-9:
            bs = s
            chosen = sel
    acc = np.zeros(Xe.shape[0])
    used = 0
    for i in chosen:
        kind, p = specs[i]
        try:
            mdl = _build(kind, p, seed)
            mdl.fit(X, y)
            pr = np.asarray(mdl.predict(Xe), dtype=float)
        except Exception:
            continue
        if pr.shape[0] == Xe.shape[0] and np.all(np.isfinite(pr)):
            acc = acc + pr
            used += 1
    if used == 0:
        return fallback
    out = acc / float(used)
    lo = float(np.min(y)) - 1.0
    hi = float(np.max(y)) + 1.0
    out = np.clip(out, lo, hi)
    if not np.all(np.isfinite(out)):
        return fallback
    return out
