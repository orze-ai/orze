import numpy as np
from sklearn.neural_network import MLPRegressor


def _feats(M):
    M = np.clip(np.asarray(M, dtype=float), 0.0, None)
    return np.hstack([M, np.sqrt(M)])


def fit_predict(train, inputs, seed):
    C = np.asarray(train["C"], dtype=float)
    y = np.asarray(train["y"], dtype=float)
    Ce = np.asarray(inputs["C"], dtype=float)
    Xtr = _feats(C)
    Xev = _feats(Ce)
    # element-set groups derived from train C only (no evaluation groups used)
    gk = {}
    gid = np.array([gk.setdefault(tuple(np.flatnonzero(r > 0).tolist()), len(gk)) for r in C])
    ng = len(gk)
    ymu = float(y.mean())
    ysd = float(y.std()) + 1e-9
    yt = (y - ymu) / ysd
    rng = np.random.RandomState(int(seed))
    preds = []
    n_seeds = 4
    max_epochs = 200
    patience = 20
    for s in range(n_seeds):
        perm = rng.permutation(ng)
        hold = np.zeros(ng, dtype=bool)
        hold[perm[: max(1, ng // 10)]] = True
        va = hold[gid]
        tr = ~va
        mlp = MLPRegressor(hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
                           alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                           max_iter=1, tol=0.0, random_state=int(seed) * 100 + s)
        best = np.inf
        best_state = None
        bad = 0
        for ep in range(max_epochs):
            mlp.partial_fit(Xtr[tr], yt[tr])
            p = mlp.predict(Xtr[va])
            rm = float(np.sqrt(np.mean((p - yt[va]) ** 2)))
            if rm < best - 1e-4:
                best = rm
                best_state = ([c.copy() for c in mlp.coefs_], [b.copy() for b in mlp.intercepts_])
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            mlp.coefs_, mlp.intercepts_ = best_state
        preds.append(mlp.predict(Xev))
    p = np.mean(np.vstack(preds), axis=0) * ysd + ymu
    p = np.clip(p, 0.0, None)
    p = np.where(np.isfinite(p), p, ymu)
    return [float(v) for v in p]
