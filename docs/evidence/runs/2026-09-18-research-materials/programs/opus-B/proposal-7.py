import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import nnls


def _hgb(seed):
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=350, learning_rate=0.07,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def _et(seed):
    return ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                               max_features=0.35, n_jobs=1,
                               random_state=seed)


def _rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_predict(train, inputs, seed=1729):
    Xtr = np.asarray(train["X"], dtype=float)
    Ctr = np.asarray(train["C"], dtype=float)
    y = np.asarray(train["y"], dtype=float)
    groups = np.asarray([str(v) for v in train["groups"]])
    Xev = np.asarray(inputs["X"], dtype=float)
    Cev = np.asarray(inputs["C"], dtype=float)
    els = [str(e) for e in inputs["elements"]]
    ymax = float(np.max(y))

    def idx(sym):
        return els.index(sym) if sym in els else -1

    iCu, iO, iFe = idx("Cu"), idx("O"), idx("Fe")

    def chem(C):
        m = C.shape[0]
        cup = np.zeros(m, dtype=bool)
        fe = np.zeros(m, dtype=bool)
        if iCu >= 0 and iO >= 0:
            cup = (C[:, iCu] > 0) & (C[:, iO] > 0)
        if iFe >= 0:
            fe = (C[:, iFe] > 0) & (~cup)
        return cup, fe

    reps_tr = {"X": Xtr, "C": Ctr, "XC": np.hstack([Xtr, Ctr])}
    reps_ev = {"X": Xev, "C": Cev, "XC": np.hstack([Xev, Cev])}
    arms = [("hgb_X", "X", "hgb"), ("hgb_C", "C", "hgb"),
            ("hgb_XC", "XC", "hgb"), ("et_XC", "XC", "et")]
    names = [a[0] for a in arms]

    n = Xtr.shape[0]
    k = len(arms)
    oof = np.zeros((n, k))
    splits = list(GroupKFold(n_splits=4).split(Xtr, y, groups=groups))
    for j in range(k):
        A = reps_tr[arms[j][1]]
        kind = arms[j][2]
        for tr_i, te_i in splits:
            mdl = _hgb(1729) if kind == "hgb" else _et(1729)
            mdl.fit(A[tr_i], y[tr_i])
            oof[te_i, j] = np.clip(mdl.predict(A[te_i]), 0.0, ymax)

    w, _ = nnls(oof, y)
    blend_oof = np.clip(oof.dot(w), 0.0, ymax)
    arm_rmse = [_rmse(oof[:, j], y) for j in range(k)]
    arm_mae = [float(np.mean(np.abs(oof[:, j] - y))) for j in range(k)]
    best_j = int(np.argmin(arm_rmse))
    blend_rmse = _rmse(blend_oof, y)
    use_blend = bool(blend_rmse < arm_rmse[best_j] - 0.02)

    cup_t, fe_t = chem(Ctr)
    oth_t = ~(cup_t | fe_t)
    by_chem = {}
    for sname, mask in (("cuprate", cup_t), ("fe_based", fe_t),
                        ("other", oth_t)):
        cnt = int(mask.sum())
        if cnt > 0:
            by_chem[sname] = {
                "n": cnt,
                "mean_y": round(float(np.mean(y[mask])), 3),
                "oof_rmse_by_arm": [round(_rmse(oof[mask, j], y[mask]), 4)
                                     for j in range(k)],
                "oof_rmse_blend": round(_rmse(blend_oof[mask], y[mask]), 4)}

    preds = np.zeros((Xev.shape[0], k))
    for j in range(k):
        A = reps_tr[arms[j][1]]
        B = reps_ev[arms[j][1]]
        if arms[j][2] == "hgb":
            acc = np.zeros(B.shape[0], dtype=float)
            for s in (1729, 7):
                m = _hgb(s)
                m.fit(A, y)
                acc += m.predict(B)
            preds[:, j] = acc / 2.0
        else:
            m = _et(1729)
            m.fit(A, y)
            preds[:, j] = m.predict(B)
        preds[:, j] = np.clip(preds[:, j], 0.0, ymax)

    if use_blend:
        pred = np.clip(preds.dot(w), 0.0, ymax)
        chosen = "nnls_blend"
    else:
        pred = preds[:, best_j]
        chosen = names[best_j]

    cup_e, fe_e = chem(Cev)
    findings = {
        "design": ("only manipulated factor is the input representation: "
                   "hgb_X=81 descriptors, hgb_C=86 stoichiometric fractions, "
                   "hgb_XC=concatenation, et_XC=ExtraTrees on concatenation. "
                   "Learner hyperparameters, GroupKFold(4) folds over train "
                   "element-set groups and CV seed 1729 identical across the "
                   "three HistGB arms; et_XC additionally changes the learner "
                   "and is therefore a diversity arm, not a clean "
                   "representation contrast."),
        "arms": names,
        "oof_rmse_by_arm": [round(v, 4) for v in arm_rmse],
        "oof_mae_by_arm": [round(v, 4) for v in arm_mae],
        "nnls_weights": [round(float(v), 4) for v in w],
        "oof_rmse_blend": round(blend_rmse, 4),
        "best_single_arm": names[best_j],
        "blend_gain_over_best_single_K": round(arm_rmse[best_j] - blend_rmse, 4),
        "decision_rule_predeclared": ("use the NNLS blend only if its train "
                                      "group-OOF RMSE beats the best single "
                                      "arm by more than 0.02 K, else emit the "
                                      "best single arm prediction"),
        "submitted_predictor": chosen,
        "by_chemistry_train_oof": by_chem,
        "eval_pool": {
            "rows": int(Xev.shape[0]),
            "pred_mean": round(float(np.mean(pred)), 3),
            "pred_p90": round(float(np.percentile(pred, 90)), 3),
            "pred_max": round(float(np.max(pred)), 3),
            "cuprate_rows": int(cup_e.sum()),
            "fe_rows": int(fe_e.sum()),
            "train_y_mean": round(float(np.mean(y)), 3),
            "train_y_max": round(ymax, 3)},
        "transductive_note": ("evaluation features enter only through frozen "
                              "models fitted on train; no evaluation labels, "
                              "groups, row ids or formulas are used and no "
                              "statistic is pooled across evaluation rows. "
                              "Element symbols come from inputs['elements'] "
                              "only to locate Cu/O/Fe columns for descriptive "
                              "subset counts."),
        "interpretation": ("All numbers above are train-only out-of-fold over "
                           "element-set groups plus descriptive statistics of "
                           "the prediction vector; they are NOT evaluation "
                           "label diagnostics. Only the separately measured "
                           "development RMSE is out-of-sample evidence. A "
                           "non-trivial NNLS weight on hgb_C with a blend OOF "
                           "gain supports complementary composition-space "
                           "information; weight concentrated on hgb_XC with "
                           "no gain supports X subsuming C. An OOF blend gain "
                           "that does not reproduce on development is a "
                           "transfer counterexample.")}
    return {"prediction": [float(v) for v in pred], "findings": findings}
