import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor

def fit_predict(train, inputs, seed):
    X = np.asarray(train['X'], dtype=float)
    C = np.asarray(train['C'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xe = np.asarray(inputs['X'], dtype=float)
    Ce = np.asarray(inputs['C'], dtype=float)
    F = np.hstack([X, C])
    Fe = np.hstack([Xe, Ce])
    # label-independent element-set ids from support pattern of C
    uniq = {}
    gid = np.empty(C.shape[0], dtype=int)
    for i in range(C.shape[0]):
        k = tuple(np.nonzero(C[i] > 0)[0].tolist())
        gid[i] = uniq.setdefault(k, len(uniq))
    G = len(uniq)
    rng = np.random.RandomState(int(seed) + 17)
    K = 8
    hgb_pred = np.zeros(Fe.shape[0])
    for k in range(K):
        draw = rng.randint(0, G, size=G)
        cnt = np.bincount(draw, minlength=G).astype(float)
        w = cnt[gid]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=int(seed) + k)
        m.fit(F, y, sample_weight=w)
        hgb_pred += m.predict(Fe) / K
    et = ExtraTreesRegressor(n_estimators=200, max_features=0.5, min_samples_leaf=2,
                             n_jobs=1, random_state=int(seed) + 101)
    et.fit(F, y)
    et_pred = et.predict(Fe)
    pred = 0.6 * hgb_pred + 0.4 * et_pred
    pred = np.clip(pred, 0.0, None)
    pred = np.where(np.isfinite(pred), pred, float(np.median(y)))
    return [float(v) for v in pred]
