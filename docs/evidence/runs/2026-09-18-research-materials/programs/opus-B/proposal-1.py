import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

FAM = [
    ("alkali", ["Li", "Na", "K", "Rb", "Cs"]),
    ("alkaline_earth", ["Be", "Mg", "Ca", "Sr", "Ba"]),
    ("rare_earth", ["Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]),
    ("actinide", ["Ac", "Th", "Pa", "U", "Np", "Pu"]),
    ("tm3d", ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]),
    ("tm4d", ["Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"]),
    ("tm5d", ["Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"]),
    ("chalcogen", ["O", "S", "Se", "Te"]),
    ("halogen", ["F", "Cl", "Br", "I"]),
    ("pnictogen", ["N", "P", "As", "Sb", "Bi"]),
    ("group14", ["C", "Si", "Ge", "Sn", "Pb"]),
    ("group13", ["B", "Al", "Ga", "In", "Tl"]),
    ("hydrogen", ["H"]),
    ("noble", ["He", "Ne", "Ar", "Kr", "Xe"]),
]

TOL = 1e-12


def _blocks(Xin, Cin, elements):
    X = np.asarray(Xin, dtype=float)
    C = np.asarray(Cin, dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    C = np.where(np.isfinite(C), C, 0.0)
    n = C.shape[0]
    m = C.shape[1]
    pres = (C > TOL).astype(float)
    idx = {}
    for i, e in enumerate(elements):
        idx[str(e)] = i
    fam_frac = {}
    fam_cnt = {}
    for name, els in FAM:
        ii = [idx[e] for e in els if e in idx and idx[e] < m]
        if len(ii) > 0:
            fam_frac[name] = C[:, ii].sum(axis=1)
            fam_cnt[name] = pres[:, ii].sum(axis=1)
        else:
            fam_frac[name] = np.zeros(n)
            fam_cnt[name] = np.zeros(n)
    feats = []
    names = []
    for name, _els in FAM:
        feats.append(fam_frac[name])
        names.append("frac_" + name)
        feats.append(fam_cnt[name])
        names.append("cnt_" + name)
    anion = fam_frac["chalcogen"] + fam_frac["halogen"] + fam_frac["pnictogen"]
    feats.append(anion)
    names.append("frac_anion")
    feats.append(anion / np.maximum(1e-9, 1.0 - anion))
    names.append("anion_ratio")
    tm = fam_frac["tm3d"] + fam_frac["tm4d"] + fam_frac["tm5d"]
    feats.append(tm)
    names.append("frac_tm_all")
    feats.append(pres.sum(axis=1))
    names.append("n_elements_from_C")
    Cs = -np.sort(-C, axis=1)
    feats.append(Cs[:, 0])
    names.append("max_frac")
    if m > 1:
        feats.append(Cs[:, 1])
    else:
        feats.append(np.zeros(n))
    names.append("second_frac")
    if m > 2:
        feats.append(Cs[:, 2])
    else:
        feats.append(np.zeros(n))
    names.append("third_frac")
    safe = np.where(C > TOL, C, 1.0)
    ent = -(np.where(C > TOL, C, 0.0) * np.log(safe)).sum(axis=1)
    feats.append(ent)
    names.append("comp_entropy")
    feats.append(safe.min(axis=1))
    names.append("min_nonzero_frac")
    F = np.column_stack(feats)
    F = np.where(np.isfinite(F), F, 0.0)
    return X, C, F, names


def _mk(seed_val):
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.07,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=int(seed_val) % 2147483647,
    )


def _rmse(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _group_mae(y, p, groups):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    vals = []
    for g in np.unique(groups):
        msk = groups == g
        vals.append(float(np.mean(np.abs(y[msk] - p[msk]))))
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def fit_predict(train, inputs, seed):
    try:
        sd = int(seed)
    except Exception:
        sd = 1729
    Xt, Ct, Ft, fam_names = _blocks(train["X"], train["C"], train["elements"])
    Xe, Ce, Fe, _ = _blocks(inputs["X"], inputs["C"], inputs["elements"])
    y = np.asarray(train["y"], dtype=float)
    groups = np.asarray([str(g) for g in train["groups"]])
    n_tr = Xt.shape[0]

    tr_blocks = {
        "X": Xt,
        "XC": np.hstack([Xt, Ct]),
        "XF": np.hstack([Xt, Ft]),
        "XCF": np.hstack([Xt, Ct, Ft]),
    }
    ev_blocks = {
        "X": Xe,
        "XC": np.hstack([Xe, Ce]),
        "XF": np.hstack([Xe, Fe]),
        "XCF": np.hstack([Xe, Ce, Fe]),
    }
    variants = [
        ("V1_X_raw", "X", False),
        ("V2_XC_raw", "XC", False),
        ("V3_XFam_raw", "XF", False),
        ("V4_XCFam_raw", "XCF", False),
        ("V5_XCFam_log", "XCF", True),
    ]

    n_groups = int(len(np.unique(groups)))
    n_splits = 4
    if n_groups < n_splits:
        n_splits = max(2, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    oof = {}
    for name, _b, _l in variants:
        oof[name] = np.zeros(n_tr)
    folds = list(gkf.split(tr_blocks["X"], y, groups))
    for fi, (itr, iva) in enumerate(folds):
        for name, bkey, use_log in variants:
            A = tr_blocks[bkey]
            mdl = _mk(sd + 17 * fi + 3)
            if use_log:
                mdl.fit(A[itr], np.log1p(np.maximum(y[itr], 0.0)))
                pv = np.expm1(mdl.predict(A[iva]))
            else:
                mdl.fit(A[itr], y[itr])
                pv = mdl.predict(A[iva])
            oof[name][iva] = np.maximum(pv, 0.0)

    per_variant = {}
    for name, _b, _l in variants:
        per_variant[name] = {
            "cv_rmse": round(_rmse(y, oof[name]), 4),
            "cv_mae": round(_mae(y, oof[name]), 4),
            "cv_group_mae": round(_group_mae(y, oof[name], groups), 4),
        }

    names = [v[0] for v in variants]
    combos = []
    for a in names:
        combos.append([a])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            combos.append([names[i], names[j]])
    best = None
    combo_scores = {}
    for cmb in combos:
        pr = np.mean(np.column_stack([oof[c] for c in cmb]), axis=1)
        r = _rmse(y, pr)
        combo_scores["+".join(cmb)] = round(r, 4)
        if best is None or r < best[0]:
            best = (r, cmb)
    chosen = best[1]

    preds = []
    for name in chosen:
        bkey = None
        use_log = False
        for nm, bk, lg in variants:
            if nm == name:
                bkey = bk
                use_log = lg
        mdl = _mk(sd + 101)
        A = tr_blocks[bkey]
        if use_log:
            mdl.fit(A, np.log1p(np.maximum(y, 0.0)))
            pe = np.expm1(mdl.predict(ev_blocks[bkey]))
        else:
            mdl.fit(A, y)
            pe = mdl.predict(ev_blocks[bkey])
        preds.append(np.maximum(pe, 0.0))
    pred = np.mean(np.column_stack(preds), axis=1)
    pred = np.clip(pred, 0.0, 200.0)

    d_id = per_variant["V2_XC_raw"]["cv_rmse"] - per_variant["V1_X_raw"]["cv_rmse"]
    d_fam = per_variant["V3_XFam_raw"]["cv_rmse"] - per_variant["V1_X_raw"]["cv_rmse"]
    d_both = per_variant["V4_XCFam_raw"]["cv_rmse"] - per_variant["V2_XC_raw"]["cv_rmse"]
    d_log = per_variant["V5_XCFam_log"]["cv_rmse"] - per_variant["V4_XCFam_raw"]["cv_rmse"]
    d_log_mae = per_variant["V5_XCFam_log"]["cv_mae"] - per_variant["V4_XCFam_raw"]["cv_mae"]

    findings = {
        "design": "train-only GroupKFold over element-set groups; identical HistGB settings across feature-block and target-transform variants",
        "n_train_rows": int(n_tr),
        "n_train_groups": n_groups,
        "n_cv_folds": int(n_splits),
        "n_features": {k: int(v.shape[1]) for k, v in tr_blocks.items()},
        "family_block_size": int(Ft.shape[1]),
        "calculated_cv": per_variant,
        "calculated_combo_cv_rmse": combo_scores,
        "chosen_for_submission": chosen,
        "calculated_contrasts_cv_rmse_delta_negative_is_better": {
            "add_element_identity_C_to_X": round(float(d_id), 4),
            "add_family_aggregates_to_X": round(float(d_fam), 4),
            "add_family_on_top_of_X_plus_C": round(float(d_both), 4),
            "log1p_target_vs_raw_same_block_rmse": round(float(d_log), 4),
            "log1p_target_vs_raw_same_block_mae": round(float(d_log_mae), 4),
        },
        "train_target_stats": {
            "mean": round(float(np.mean(y)), 3),
            "median": round(float(np.median(y)), 3),
            "p90": round(float(np.percentile(y, 90)), 3),
            "max": round(float(np.max(y)), 3),
        },
        "prediction_stats": {
            "mean": round(float(np.mean(pred)), 3),
            "median": round(float(np.median(pred)), 3),
            "p90": round(float(np.percentile(pred, 90)), 3),
            "max": round(float(np.max(pred)), 3),
            "frac_below_5K": round(float(np.mean(pred < 5.0)), 4),
        },
        "acknowledged_transformations": "evaluation features transformed only by deterministic per-row functions of X and C (chemical-family sums/counts, entropy, sorted-fraction order statistics); no label or cross-row fitting on evaluation data",
        "interpretation_unverified": "CV deltas are train-internal and grouped by element set; they do not by themselves establish development or confirmation behaviour, and the combination choice is a joint exploratory selection rather than a clean per-component attribution.",
    }
    return {"prediction": [float(v) for v in pred], "findings": findings}
