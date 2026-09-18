import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

RE_LIST = ['Sc','Y','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']

def _stoich(C, elements):
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    idx = {e: i for i, e in enumerate(elements)}
    def f(names):
        cols = [idx[x] for x in names if x in idx]
        return C[:, cols].sum(axis=1) if cols else np.zeros(n)
    def ratio(a, b):
        return np.where(b > 0, np.minimum(a / np.maximum(b, 1e-9), 50.0), 0.0)
    O = f(['O']); Cu = f(['Cu']); Ca = f(['Ca']); Ba = f(['Ba']); Sr = f(['Sr'])
    AE = Ba + Sr + Ca
    RE = f(RE_LIST)
    HV = f(['Tl','Bi','Hg','Pb'])
    Fe = f(['Fe']); Pn = f(['As','P','Se','Te','S'])
    B = f(['B']); Mg = f(['Mg']); Nb = f(['Nb'])
    alk = f(['Li','Na','K','Rb','Cs'])
    hal = f(['F','Cl'])
    cat = np.clip(1.0 - O - hal, 0.0, 1.0)
    nz = (C > 0).sum(axis=1).astype(float)
    cmax = C.max(axis=1)
    Cp = np.clip(C, 1e-12, None)
    ent = -(np.where(C > 0, C * np.log(Cp), 0.0)).sum(axis=1)
    cols = [O, Cu, Ca, Ba, Sr, AE, RE, HV, Fe, Pn, B, Mg, Nb, alk, hal, cat,
            ratio(O, Cu), ratio(Ca, Cu), ratio(AE, Cu), ratio(RE, Cu), ratio(HV, Cu),
            ratio(O, cat), ratio(Pn, Fe), ratio(B, Mg), ratio(Sr, AE), ratio(Ca, AE), ratio(Ba, AE),
            nz, cmax, ent,
            ((Cu > 0) & (O > 0)).astype(float), ((Fe > 0) & (Pn > 0)).astype(float), ((Mg > 0) & (B > 0)).astype(float)]
    return np.column_stack([np.asarray(c, dtype=float) for c in cols])

def _model(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)

def _rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0:
        return None
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _bias(pred, y, mask):
    if not np.any(mask):
        return None
    return float(np.mean(pred[mask] - y[mask]))

def fit_predict(train, inputs, seed):
    Xtr = np.asarray(train['X'], float); Ctr = np.asarray(train['C'], float); ytr = np.asarray(train['y'], float)
    Xev = np.asarray(inputs['X'], float); Cev = np.asarray(inputs['C'], float)
    els = list(train['elements'])
    Str = _stoich(Ctr, els)
    Sev = _stoich(Cev, list(inputs['elements']))
    base_tr = np.hstack([Xtr, Ctr])
    ext_tr = np.hstack([Xtr, Ctr, Str])
    ext_ev = np.hstack([Xev, Cev, Sev])
    m = _model(1729).fit(ext_tr, ytr)
    pred = np.clip(m.predict(ext_ev), 0.0, None)
    findings = {'variant_deployed': 'HGB(250 iters, lr 0.08, 31 leaves, l2=1, seed 1729) on X+C+stoichiometric ratio features; target untransformed; predictions clipped at 0',
                'n_stoich_features': int(Str.shape[1]), 'n_train_rows': int(len(ytr)), 'n_eval_rows': int(len(pred)),
                'eval_pred_mean': float(np.mean(pred)), 'eval_pred_frac_gt40': float(np.mean(pred > 40))}
    try:
        groups = np.asarray(train['groups'])
        gkf = GroupKFold(n_splits=5)
        if 'Cu' in els and 'O' in els:
            cup = (Ctr[:, els.index('Cu')] > 0) & (Ctr[:, els.index('O')] > 0)
        else:
            cup = np.zeros(len(ytr), bool)
        res = {}
        variants = [('X+C', base_tr, False), ('X+C+stoich', ext_tr, False), ('X+C+stoich_sqrt_target', ext_tr, True)]
        for name, F, sq in variants:
            oof = np.zeros(len(ytr))
            for tr_i, te_i in gkf.split(F, ytr, groups):
                yt = np.sqrt(np.clip(ytr[tr_i], 0, None)) if sq else ytr[tr_i]
                mm = _model(1729).fit(F[tr_i], yt)
                p = mm.predict(F[te_i])
                oof[te_i] = np.clip(p, 0, None) ** 2 if sq else p
            res[name] = {'cv_rmse': _rmse(ytr, oof), 'cv_mae': float(np.mean(np.abs(ytr - oof))),
                         'cv_rmse_cuprate': _rmse(ytr[cup], oof[cup]), 'cv_rmse_noncuprate': _rmse(ytr[~cup], oof[~cup]),
                         'bias_y_gt40': _bias(oof, ytr, ytr > 40), 'bias_y_lt10': _bias(oof, ytr, ytr < 10)}
        findings['train_groupkfold5_by_element_set'] = res
        findings['n_train_cuprate_rows'] = int(cup.sum())
        findings['note'] = 'CV is on train only (GroupKFold by element set), not the development score. Hyperparameters identical across variants; only feature set (and, for the third variant, target transform) differs. Deployed prediction uses the X+C+stoich untransformed variant regardless of CV outcome.'
    except Exception as e:
        findings['cv_error'] = str(e)[:200]
    return {'prediction': [float(v) for v in pred], 'findings': findings}
