import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

SMOOTH = 5.0
K_MODELS = 8
N_FOLDS = 5


def _set_keys(C):
    keys = {}
    ids = np.empty(C.shape[0], dtype=int)
    for i in range(C.shape[0]):
        k = tuple(np.flatnonzero(C[i] > 0).tolist())
        if k not in keys:
            keys[k] = len(keys)
        ids[i] = keys[k]
    return ids


def _stats(C, y):
    n, E = C.shape
    present = C > 0
    ly = np.log1p(np.maximum(y, 0.0))
    g_mean = float(y.mean())
    g_lmean = float(ly.mean())
    g_q90 = float(np.quantile(y, 0.9))
    g_std = float(y.std())
    P = present.astype(float)
    cnt = P.sum(axis=0)
    mean = (P.T @ y + SMOOTH * g_mean) / (cnt + SMOOTH)
    lmean = (P.T @ ly + SMOOTH * g_lmean) / (cnt + SMOOTH)
    q90 = np.full(E, g_q90)
    std = np.full(E, g_std)
    for j in range(E):
        if cnt[j] >= 3:
            yj = y[present[:, j]]
            q90[j] = (cnt[j] * float(np.quantile(yj, 0.9)) + SMOOTH * g_q90) / (cnt[j] + SMOOTH)
            std[j] = (cnt[j] * float(yj.std()) + SMOOTH * g_std) / (cnt[j] + SMOOTH)
    return np.vstack([mean, lmean, q90, std, np.log1p(cnt)])


def _encode(C, S):
    present = C > 0
    has = present.any(axis=1)
    cols = []
    for k in range(4):
        e = S[k]
        wm = C @ e
        mx = np.where(present, e[None, :], -np.inf).max(axis=1)
        mn = np.where(present, e[None, :], np.inf).min(axis=1)
        mx = np.where(has, mx, wm)
        mn = np.where(has, mn, wm)
        cols += [wm, mx, mn]
    lc = S[4]
    wc = C @ lc
    mc = np.where(present, lc[None, :], np.inf).min(axis=1)
    mc = np.where(has, mc, wc)
    cols += [wc, mc]
    return np.column_stack(cols)


def fit_predict(train, inputs, seed):
    seed = int(seed) if seed is not None else 0
    rng = np.random.RandomState(seed)
    C = np.nan_to_num(np.asarray(train['C'], dtype=float))
    y = np.asarray(train['y'], dtype=float)
    Ce = np.nan_to_num(np.asarray(inputs['C'], dtype=float))
    sid = _set_keys(C)
    n_sets = int(sid.max()) + 1
    perm = rng.permutation(n_sets)
    fold_of_set = np.empty(n_sets, dtype=int)
    fold_of_set[perm] = np.arange(n_sets) % N_FOLDS
    fold = fold_of_set[sid]
    enc_tr = np.zeros((C.shape[0], 14))
    for f in range(N_FOLDS):
        m = fold == f
        if m.sum() == 0:
            continue
        S = _stats(C[~m], y[~m])
        enc_tr[m] = _encode(C[m], S)
    S_full = _stats(C, y)
    enc_ev = _encode(Ce, S_full)
    Xtr = np.hstack([C, enc_tr])
    Xev = np.hstack([Ce, enc_ev])
    preds = np.zeros(Ce.shape[0])
    for k in range(K_MODELS):
        draw = rng.randint(0, n_sets, size=n_sets)
        cnt_set = np.bincount(draw, minlength=n_sets).astype(float)
        w = cnt_set[sid]
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False, random_state=seed + 1000 + k)
        model.fit(Xtr, y, sample_weight=w)
        preds += model.predict(Xev)
    preds /= K_MODELS
    preds = np.clip(preds, 0.0, None)
    preds = np.nan_to_num(preds, nan=float(y.mean()), posinf=float(y.max()), neginf=0.0)
    return [float(v) for v in preds]
