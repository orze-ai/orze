import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

EPS = 1e-9
KEYS = ("wmean", "pmean", "pmax", "p75")
CFG = {
    "A_X_log": (False, False, True),
    "B_XC_log": (True, False, True),
    "C_XCE_log": (True, True, True),
    "D_XCE_raw": (True, True, False),
}


def _arr(a):
    return np.asarray(a, dtype=float)


def _clean(a):
    return np.nan_to_num(_arr(a), nan=0.0, posinf=0.0, neginf=0.0)


def _norm_C(C):
    P = np.clip(_clean(C), 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    return P / np.maximum(s, 1e-12)


def _groups_from_C(C):
    return np.array(["".join("1" if v > EPS else "0" for v in r) for r in C])


def _comp_feats(C):
    P = np.where(C > EPS, C, 0.0)
    pres = P > 0
    n_el = pres.sum(axis=1).astype(float)
    srt = -np.sort(-P, axis=1)[:, :5]
    lg = np.where(P > 0, np.log(np.maximum(P, 1e-12)), 0.0)
    ent = -(P * lg).sum(axis=1)
    dop = np.where(pres & (P < 0.06), P, 0.0).sum(axis=1)
    return np.column_stack([n_el, ent, dop, srt, srt[:, 0] - srt[:, 1]])


def _elem_stats(C, ly):
    n_el = C.shape[1]
    P = np.where(C > EPS, C, 0.0)
    pres = P > 0
    out = {k: np.full(n_el, np.nan) for k in KEYS}
    cnt = np.zeros(n_el)
    for e in range(n_el):
        m = pres[:, e]
        c = int(m.sum())
        cnt[e] = float(c)
        if c > 0:
            v = ly[m]
            w = P[m, e]
            out["wmean"][e] = float((w * v).sum() / max(float(w.sum()), 1e-12))
            out["pmean"][e] = float(v.mean())
            out["pmax"][e] = float(v.max())
            out["p75"][e] = float(np.percentile(v, 75))
    out["cnt"] = cnt
    return out


def _enc_feats(C, st, gmean):
    P = np.where(C > EPS, C, 0.0)
    pres = P > 0
    wsum = P.sum(axis=1)
    dom = np.argmax(P, axis=1)
    cols = []
    for k in KEYS:
        v = np.asarray(st[k], dtype=float)
        known = np.isfinite(v)
        V = np.where(known, v, 0.0)
        M = pres & known[None, :]
        W = P * M
        wd = W.sum(axis=1)
        wm = np.where(wd > 0, (W * V[None, :]).sum(axis=1) / np.maximum(wd, 1e-12), gmean)
        c = M.sum(axis=1)
        um = np.where(c > 0, (M * V[None, :]).sum(axis=1) / np.maximum(c, 1), gmean)
        mx = np.where(M, V[None, :], -1e18).max(axis=1)
        mx = np.where(mx > -1e17, mx, gmean)
        mn = np.where(M, V[None, :], 1e18).min(axis=1)
        mn = np.where(mn < 1e17, mn, gmean)
        dv = np.where(known, v, gmean)[dom]
        cols += [wm, um, mx, mn, dv, mx - mn]
    cnt = np.asarray(st["cnt"], dtype=float)
    lc = np.log1p(cnt)
    seen = cnt > 0
    n_uns = (pres & (~seen)[None, :]).sum(axis=1).astype(float)
    f_uns = (P * (~seen)[None, :].astype(float)).sum(axis=1)
    mlc = np.where(pres, lc[None, :], 1e18).min(axis=1)
    mlc = np.where(mlc < 1e17, mlc, 0.0)
    wlc = (P * lc[None, :]).sum(axis=1) / np.maximum(wsum, 1e-12)
    cols += [n_uns, f_uns, mlc, wlc]
    return np.column_stack(cols)


def _oof_enc(C, ly, g, n_splits=5):
    n = C.shape[0]
    uniq = len(set(g.tolist()))
    k = int(max(2, min(n_splits, uniq)))
    F = None
    for tr, va in GroupKFold(n_splits=k).split(C, ly, groups=g):
        st = _elem_stats(C[tr], ly[tr])
        f = _enc_feats(C[va], st, float(ly[tr].mean()))
        if F is None:
            F = np.zeros((n, f.shape[1]))
        F[va] = f
    return F


def _design(X, comp, C, E, use_C, use_E):
    parts = [X, comp]
    if use_C:
        parts.append(C)
    if use_E:
        parts.append(E)
    return np.column_stack(parts)


def _hgb(rs):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.07,
                                        max_leaf_nodes=31, min_samples_leaf=20,
                                        l2_regularization=1.0,
                                        early_stopping=False, random_state=int(rs))


def _inv(p, log_t, ymax):
    if log_t:
        p = np.expm1(np.clip(p, 0.0, 12.0))
    return np.clip(p, 0.0, ymax)


def fit_predict(train, inputs, seed):
    rs = int(seed) if seed is not None else 0
    Xtr = _clean(train["X"])
    Xev = _clean(inputs["X"])
    Ctr = _norm_C(train["C"])
    Cev = _norm_C(inputs["C"])
    y = np.clip(_clean(train["y"]).ravel(), 0.0, None)
    ly = np.log1p(y)
    ymax = float(y.max()) * 1.02 + 1.0
    med = float(np.median(y))
    try:
        g = np.array([str(v) for v in train["groups"]])
        if g.shape[0] != Xtr.shape[0]:
            raise ValueError("bad groups")
    except Exception:
        g = _groups_from_C(Ctr)
    comp_tr = _comp_feats(Ctr)
    comp_ev = _comp_feats(Cev)
    names = list(CFG.keys())
    chosen = ["C_XCE_log"]
    try:
        oof = {nm: np.full(y.shape[0], np.nan) for nm in names}
        folds = list(GroupKFold(n_splits=5).split(Xtr, y, groups=g))
        for tr, va in folds:
            Etr = _oof_enc(Ctr[tr], ly[tr], g[tr])
            st = _elem_stats(Ctr[tr], ly[tr])
            Eva = _enc_feats(Ctr[va], st, float(ly[tr].mean()))
            for nm in names:
                uc, ue, lt = CFG[nm]
                A = _design(Xtr[tr], comp_tr[tr], Ctr[tr], Etr, uc, ue)
                B = _design(Xtr[va], comp_tr[va], Ctr[va], Eva, uc, ue)
                t = ly[tr] if lt else y[tr]
                m = _hgb(rs + 11)
                m.fit(A, t)
                oof[nm][va] = _inv(m.predict(B), lt, ymax)
        cands = []
        for i, a in enumerate(names):
            pa = oof[a]
            if np.all(np.isfinite(pa)):
                cands.append(([a], float(np.sqrt(np.mean((pa - y) ** 2)))))
            for b in names[i + 1:]:
                pb = oof[b]
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    mix = 0.5 * (pa + pb)
                    cands.append(([a, b], float(np.sqrt(np.mean((mix - y) ** 2)))))
        if cands:
            cands.sort(key=lambda z: z[1])
            chosen = cands[0][0]
    except Exception:
        chosen = ["C_XCE_log"]
    try:
        Eall = _oof_enc(Ctr, ly, g)
        st_all = _elem_stats(Ctr, ly)
        Eev = _enc_feats(Cev, st_all, float(ly.mean()))
        acc = []
        for nm in chosen:
            uc, ue, lt = CFG[nm]
            A = _design(Xtr, comp_tr, Ctr, Eall, uc, ue)
            B = _design(Xev, comp_ev, Cev, Eev, uc, ue)
            t = ly if lt else y
            ps = []
            for s in (rs + 1, rs + 2, rs + 3):
                m = _hgb(s)
                m.fit(A, t)
                ps.append(_inv(m.predict(B), lt, ymax))
            acc.append(np.mean(np.vstack(ps), axis=0))
        pred = np.mean(np.vstack(acc), axis=0)
    except Exception:
        m = _hgb(rs + 7)
        m.fit(np.column_stack([Xtr, comp_tr]), ly)
        pred = _inv(m.predict(np.column_stack([Xev, comp_ev])), True, ymax)
    pred = np.asarray(pred, dtype=float).ravel()
    pred = np.where(np.isfinite(pred), pred, med)
    pred = np.clip(pred, 0.0, ymax)
    if pred.shape[0] != np.asarray(inputs["X"]).shape[0]:
        pred = np.full(np.asarray(inputs["X"]).shape[0], med, dtype=float)
    return [float(v) for v in pred]
