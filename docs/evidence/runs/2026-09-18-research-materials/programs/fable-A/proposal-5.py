import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def _mat(m):
    C = np.asarray(m["C"], dtype=float)
    X = np.asarray(m["X"], dtype=float)
    return np.hstack([C, X])


def fit_predict(train, inputs, seed):
    Xtr = _mat(train)
    ytr = np.asarray(train["y"], dtype=float)
    Xev = _mat(inputs)
    Ctr = np.asarray(train["C"], dtype=float)
    # label-independent element-set keys from composition support
    keys = {}
    for i in range(Ctr.shape[0]):
        k = tuple(np.flatnonzero(Ctr[i] > 0).tolist())
        keys.setdefault(k, []).append(i)
    sets = [np.asarray(v, dtype=int) for v in keys.values()]
    n_sets = len(sets)
    try:
        s = int(seed)
    except Exception:
        s = 0
    rng = np.random.RandomState(s)
    K = 8
    preds = np.zeros(Xev.shape[0], dtype=float)
    med = float(np.median(ytr))
    for k in range(K):
        draws = rng.randint(0, n_sets, size=n_sets)
        counts = np.bincount(draws, minlength=n_sets)
        w = np.zeros(Xtr.shape[0], dtype=float)
        for j in range(n_sets):
            if counts[j] > 0:
                w[sets[j]] = float(counts[j])
        mask = w > 0
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=1729 + k,
        )
        model.fit(Xtr[mask], ytr[mask], sample_weight=w[mask])
        preds += model.predict(Xev)
    preds /= float(K)
    preds = np.clip(preds, 0.0, None)
    preds = np.where(np.isfinite(preds), preds, med)
    return [float(v) for v in preds]
