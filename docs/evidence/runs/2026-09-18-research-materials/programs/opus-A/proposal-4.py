import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def _arr(m):
    X = np.asarray(m["X"], dtype=float)
    C = np.asarray(m["C"], dtype=float)
    y = np.asarray(m["y"], dtype=float)
    g = m.get("groups")
    if g is None:
        g = [""] * int(y.shape[0])
    g = np.asarray([str(v) for v in g], dtype=object)
    return X, C, y, g


def _keys(C, elements, tol=1e-9):
    els = [str(e) for e in elements]
    out = []
    for i in range(C.shape[0]):
        idx = np.nonzero(C[i] > tol)[0]
        out.append("-".join(els[j] for j in idx))
    return np.asarray(out, dtype=object)


def _cent(F, keys):
    uk, inv = np.unique(keys, return_inverse=True)
    s = np.zeros((uk.shape[0], F.shape[1]), dtype=float)
    c = np.zeros(uk.shape[0], dtype=float)
    Fz = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    np.add.at(s, inv, Fz)
    np.add.at(c, inv, 1.0)
    return uk, inv, s / np.maximum(c, 1.0)[:, None], c


def _gmean_y(y, keys):
    uk, inv = np.unique(keys, return_inverse=True)
    s = np.zeros(uk.shape[0], dtype=float)
    c = np.zeros(uk.shape[0], dtype=float)
    np.add.at(s, inv, y)
    np.add.at(c, inv, 1.0)
    return uk, inv, s / np.maximum(c, 1.0), c


def _shape(C):
    S = np.sort(C, axis=1)[:, ::-1]
    top = S[:, :4]
    simp = np.sum(C ** 2, axis=1, keepdims=True)
    n = np.sum(C > 1e-9, axis=1, keepdims=True).astype(float)
    r2 = np.where(top[:, 1:2] > 1e-9, top[:, 0:1] / np.maximum(top[:, 1:2], 1e-9), 0.0)
    r2 = np.minimum(r2, 1000.0)
    ent = -np.sum(np.where(C > 1e-12, C * np.log(np.maximum(C, 1e-12)), 0.0), axis=1, keepdims=True)
    return np.hstack([top, simp, n, r2, ent])


def _hgb(seed, it=200, lr=0.08):
    return HistGradientBoostingRegressor(max_iter=it, learning_rate=lr, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=int(seed) % 2147483647)


def _sc(y, p, keys):
    p = np.asarray(p, dtype=float)
    p = np.nan_to_num(p, nan=float(np.mean(y)), posinf=float(np.max(y)), neginf=0.0)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    uk, inv = np.unique(keys, return_inverse=True)
    gm = []
    for j in range(uk.shape[0]):
        m = inv == j
        gm.append(float(np.mean(np.abs(y[m] - p[m]))))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "group_mae": round(float(np.mean(gm)), 4)}


def analyze(data, history, seed):
    tr = data["train"]
    dv = data["development"]
    Xtr, Ctr, ytr, gtr = _arr(tr)
    Xdv, Cdv, ydv, gdv = _arr(dv)
    els = tr.get("elements") or dv.get("elements")
    ktr = _keys(Ctr, els)
    kdv = _keys(Cdv, els)

    sanity = {
        "train_rows": int(ytr.shape[0]),
        "dev_rows": int(ydv.shape[0]),
        "train_key_eq_groups_frac": round(float(np.mean(ktr == gtr)), 4),
        "dev_key_eq_groups_frac": round(float(np.mean(kdv == gdv)), 4),
        "n_train_keys": int(np.unique(ktr).size),
        "n_dev_keys": int(np.unique(kdv).size),
        "train_dev_key_overlap": int(np.intersect1d(np.unique(ktr), np.unique(kdv)).size),
    }

    ukd, invd, gyd, cntd = _gmean_y(ydv, kdv)
    within_dev = ydv - gyd[invd]
    sizes = cntd
    structure = {
        "dev_groups": int(ukd.size),
        "dev_groups_size1": int(np.sum(sizes == 1)),
        "dev_rows_in_multirow_groups_frac": round(float(np.sum(sizes[invd] > 1) / ydv.shape[0]), 4),
        "dev_median_group_size": float(np.median(sizes)),
        "dev_max_group_size": int(np.max(sizes)),
        "dev_total_sd_K": round(float(np.std(ydv)), 4),
        "dev_within_group_rms_K": round(float(np.sqrt(np.mean(within_dev ** 2))), 4),
        "dev_between_group_sd_K": round(float(np.sqrt(max(np.var(ydv) - np.mean(within_dev ** 2), 0.0))), 4),
    }

    Ftr = np.hstack([Xtr, Ctr, _shape(Ctr)])
    Fdv = np.hstack([Xdv, Cdv, _shape(Cdv)])

    res = {}

    m_flat = _hgb(seed, it=250).fit(Ftr, ytr)
    p_flat = m_flat.predict(Fdv)
    res["A_flat_hgb_X_C_shape"] = _sc(ydv, p_flat, kdv)

    ukt, invt, cent_tr, cntt = _cent(Ftr, ktr)
    _, _, gyt, _ = _gmean_y(ytr, ktr)
    ukd2, invd2, cent_dv, cntd2 = _cent(Fdv, kdv)
    Dtr = np.nan_to_num(Ftr, nan=0.0) - cent_tr[invt]
    Ddv = np.nan_to_num(Fdv, nan=0.0) - cent_dv[invd2]

    m_grp = _hgb(seed + 1, it=250).fit(cent_tr, gyt, sample_weight=cntt)
    g_pred_rows = m_grp.predict(cent_dv)[invd2]
    res["B0_group_model_only"] = _sc(ydv, g_pred_rows, kdv)

    m_win = _hgb(seed + 2, it=250).fit(Dtr, ytr - gyt[invt])
    w_pred = m_win.predict(Ddv)
    res["B_group_plus_within"] = _sc(ydv, g_pred_rows + w_pred, kdv)

    s = np.zeros(ukd2.shape[0], dtype=float)
    np.add.at(s, invd2, p_flat)
    flat_group_mean = (s / np.maximum(cntd2, 1.0))[invd2]
    res["B2_flatsmoothed_group_plus_within"] = _sc(ydv, flat_group_mean + w_pred, kdv)
    res["B3_flatsmoothed_group_only"] = _sc(ydv, flat_group_mean, kdv)

    res["E_flat_plus_centered_features"] = _sc(
        ydv, _hgb(seed + 3, it=250).fit(np.hstack([Ftr, Dtr]), ytr).predict(np.hstack([Fdv, Ddv])), kdv)

    res["F_blend_A_B"] = _sc(ydv, 0.5 * p_flat + 0.5 * (g_pred_rows + w_pred), kdv)

    oracle = {
        "true_groupmean_plus_zero": _sc(ydv, gyd[invd], kdv),
        "true_groupmean_plus_predicted_within": _sc(ydv, gyd[invd] + w_pred, kdv),
        "predicted_groupmean_plus_true_within": _sc(ydv, g_pred_rows + within_dev, kdv),
        "flat_groupmean_error_only_rmse": round(float(np.sqrt(np.mean((flat_group_mean - gyd[invd]) ** 2))), 4),
    }

    den = float(np.mean(within_dev ** 2))
    wr2 = 1.0 - float(np.mean((within_dev - w_pred) ** 2)) / den if den > 0 else None
    within_diag = {
        "within_target_rms_K": round(float(np.sqrt(den)), 4),
        "within_pred_rms_K": round(float(np.sqrt(np.mean(w_pred ** 2))), 4),
        "within_pred_R2": None if wr2 is None else round(wr2, 4),
        "within_pred_corr": round(float(np.corrcoef(within_dev, w_pred)[0, 1]), 4) if np.std(w_pred) > 1e-9 else 0.0,
    }

    mm = sizes[invd] > 1
    breakdown = {
        "A_rmse_multirow_groups": round(float(np.sqrt(np.mean((ydv[mm] - p_flat[mm]) ** 2))), 4) if np.any(mm) else None,
        "A_rmse_singleton_groups": round(float(np.sqrt(np.mean((ydv[~mm] - p_flat[~mm]) ** 2))), 4) if np.any(~mm) else None,
        "B_rmse_multirow_groups": round(float(np.sqrt(np.mean((ydv[mm] - (g_pred_rows + w_pred)[mm]) ** 2))), 4) if np.any(mm) else None,
        "B_rmse_singleton_groups": round(float(np.sqrt(np.mean((ydv[~mm] - (g_pred_rows + w_pred)[~mm]) ** 2))), 4) if np.any(~mm) else None,
    }

    hist = []
    try:
        for a in (history.get("actions") or []):
            hist.append({"task_id": a.get("task_id"), "kind": a.get("kind"), "valid": a.get("valid")})
    except Exception:
        hist = []

    findings = {
        "calculated": {
            "sanity": sanity,
            "dev_group_structure": structure,
            "candidates_train_to_dev": res,
            "oracle_substitutions": oracle,
            "within_model_diagnostics": within_diag,
            "error_by_group_size": breakdown,
        },
        "history_seen": hist[:8],
        "interpretation": "Internal train->development fits only; not officially measured method scores. Group keys for dev were derived from C (transductive use of the unlabeled pool's element-set structure, acknowledged). Compare B/B2/E against A in this same run to judge whether target decomposition or feature centering helps; use the oracle rows to decide whether remaining headroom is set-level ranking or within-set doping resolution.",
    }
    out = json.dumps(findings)
    if len(out.encode("utf-8")) > 8000:
        findings["history_seen"] = []
        out = json.dumps(findings)
    return json.loads(out)
