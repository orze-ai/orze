import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

VAL = {'H':1.0,'Li':1.0,'Na':1.0,'K':1.0,'Rb':1.0,'Cs':1.0,'Ag':1.0,
 'Be':2.0,'Mg':2.0,'Ca':2.0,'Sr':2.0,'Ba':2.0,'Ra':2.0,'Zn':2.0,'Cd':2.0,'Hg':2.0,
 'Cu':2.0,'Ni':2.0,'Co':2.0,'Fe':2.0,'Mn':2.0,'Pd':2.0,'Pb':2.0,'Eu':2.0,
 'Sc':3.0,'Y':3.0,'La':3.0,'Ce':3.0,'Pr':3.0,'Nd':3.0,'Pm':3.0,'Sm':3.0,'Gd':3.0,
 'Tb':3.0,'Dy':3.0,'Ho':3.0,'Er':3.0,'Tm':3.0,'Yb':3.0,'Lu':3.0,'Ac':3.0,
 'Al':3.0,'Ga':3.0,'In':3.0,'Tl':3.0,'B':3.0,'Bi':3.0,'Cr':3.0,'Rh':3.0,'Au':3.0,
 'Ti':4.0,'Zr':4.0,'Hf':4.0,'Si':4.0,'Ge':4.0,'Sn':4.0,'C':4.0,'Ru':4.0,'Os':4.0,
 'Ir':4.0,'Pt':4.0,'Re':4.0,'Th':4.0,'Pu':4.0,'Np':4.0,
 'V':5.0,'Nb':5.0,'Ta':5.0,'Pa':5.0,'Mo':6.0,'W':6.0,'U':6.0,'Tc':7.0,
 'O':-2.0,'S':-2.0,'Se':-2.0,'Te':-2.0,'Po':-2.0,'F':-1.0,'Cl':-1.0,'Br':-1.0,'I':-1.0,'At':-1.0,
 'N':-3.0,'P':-3.0,'As':-3.0,'Sb':-3.0,
 'He':0.0,'Ne':0.0,'Ar':0.0,'Kr':0.0,'Xe':0.0,'Rn':0.0}

ALK = ['Li','Na','K','Rb','Cs']
AE = ['Be','Mg','Ca','Sr','Ba']
RE = ['Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
TM3 = ['Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']
TM45 = ['Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg']
AN = ['O','S','Se','Te','F','Cl','Br','I','N','P','As','Sb']
PN = ['N','P','As','Sb']
CH = ['S','Se','Te']

ENAMES = ['net_charge','charge_per_anion','f_anion','f_cation','anion_cation_ratio','f_O','f_Cu','f_Fe','f_pnictide','f_chalcogen','f_alkali','f_alkaline_earth','f_rare_earth','f_tm3d','f_tm4d5d','cu_formal_valence','cu_hole_p','fe_formal_valence','O_over_Cu','n_elements','max_frac','min_frac','comp_entropy','is_cuprate','is_fe_based']


def _align(C, elements, ref):
    C = np.asarray(C, dtype=float)
    if list(elements) == list(ref):
        return C
    idx = {e: i for i, e in enumerate(elements)}
    out = np.zeros((C.shape[0], len(ref)))
    for j, e in enumerate(ref):
        i = idx.get(e)
        if i is not None:
            out[:, j] = C[:, i]
    return out


def _eng(C, elements):
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    idx = {e: i for i, e in enumerate(elements)}
    v = np.array([VAL.get(e, 0.0) for e in elements], dtype=float)
    qv = C * v[None, :]
    q = qv.sum(axis=1)

    def gf(sym):
        i = idx.get(sym)
        if i is None:
            return np.zeros(n)
        return C[:, i].copy()

    def gs(syms):
        s = np.zeros(n)
        for e in syms:
            s = s + gf(e)
        return s

    fO = gf('O')
    fCu = gf('Cu')
    fFe = gf('Fe')
    f_an = gs(AN)
    f_cat = np.clip(1.0 - f_an, 0.0, None)
    ratio = f_an / np.maximum(f_cat, 1e-9)
    qpa = q / np.maximum(f_an, 1e-9)
    cu_m = fCu > 1e-9
    q_rest = q - (qv[:, idx['Cu']] if 'Cu' in idx else 0.0)
    cu_val = np.where(cu_m, -q_rest / np.maximum(fCu, 1e-9), np.nan)
    p_hole = cu_val - 2.0
    fe_m = fFe > 1e-9
    q_r2 = q - (qv[:, idx['Fe']] if 'Fe' in idx else 0.0)
    fe_val = np.where(fe_m, -q_r2 / np.maximum(fFe, 1e-9), np.nan)
    o_cu = np.where(cu_m, fO / np.maximum(fCu, 1e-9), np.nan)
    pos = C > 1e-9
    nel = pos.sum(axis=1).astype(float)
    maxf = C.max(axis=1)
    minf = np.where(pos, C, 1.0).min(axis=1)
    ent = -(np.where(pos, C * np.log(np.maximum(C, 1e-12)), 0.0)).sum(axis=1)
    f_pn = gs(PN)
    f_ch = gs(CH)
    is_cup = ((fCu > 1e-9) & (fO > 1e-9)).astype(float)
    is_fe = ((fFe > 1e-9) & ((f_pn + f_ch) > 1e-9)).astype(float)
    cols = [q, qpa, f_an, f_cat, ratio, fO, fCu, fFe, f_pn, f_ch,
            gs(ALK), gs(AE), gs(RE), gs(TM3), gs(TM45),
            cu_val, p_hole, fe_val, o_cu, nel, maxf, minf, ent, is_cup, is_fe]
    E = np.column_stack(cols)
    E = np.where(np.isfinite(E) | np.isnan(E), E, np.nan)
    return E


def _mk(seed):
    return HistGradientBoostingRegressor(loss='squared_error', max_iter=350,
                                        learning_rate=0.07, max_leaf_nodes=31,
                                        min_samples_leaf=20, l2_regularization=1.0,
                                        max_bins=255, early_stopping=False,
                                        random_state=seed)


def _rmse(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def fit_predict(train, inputs, seed=0):
    els = list(train['elements'])
    Xtr = np.asarray(train['X'], dtype=float)
    Ctr = np.asarray(train['C'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    Xte = np.asarray(inputs['X'], dtype=float)
    Cte = _align(inputs['C'], list(inputs['elements']), els)

    g = train.get('groups')
    if g is None:
        g = ['-'.join([els[j] for j in np.nonzero(Ctr[i] > 1e-9)[0]])
             for i in range(Ctr.shape[0])]
    g = np.asarray([str(t) for t in g])

    Etr = _eng(Ctr, els)
    Ete = _eng(Cte, els)
    Fb = np.hstack([Xtr, Ctr])
    Fa = np.hstack([Xtr, Ctr, Etr])
    Tb = np.hstack([Xte, Cte])
    Ta = np.hstack([Xte, Cte, Ete])

    is_cup = Etr[:, ENAMES.index('is_cuprate')] > 0.5
    is_fe = (Etr[:, ENAMES.index('is_fe_based')] > 0.5) & (~is_cup)
    is_oth = (~is_cup) & (~is_fe)

    nsp = 4
    gkf = GroupKFold(n_splits=nsp)
    oof = {'base': np.zeros(len(y)), 'aug': np.zeros(len(y))}
    for tr, va in gkf.split(Fb, y, groups=g):
        for key, F in (('base', Fb), ('aug', Fa)):
            m = _mk(1729)
            m.fit(F[tr], y[tr])
            oof[key][va] = m.predict(F[va])

    def block(mask):
        return {'n': int(mask.sum()),
                'mean_y': round(float(np.mean(y[mask])), 3) if mask.sum() else None,
                'oof_rmse_base': round(_rmse(y[mask], oof['base'][mask]), 4) if mask.sum() else None,
                'oof_rmse_aug': round(_rmse(y[mask], oof['aug'][mask]), 4) if mask.sum() else None}

    ymax = float(np.max(y))
    preds = np.zeros(Ta.shape[0])
    seeds = [1729, 7, 20260918]
    for s in seeds:
        m = _mk(s)
        m.fit(Fa, y)
        preds += m.predict(Ta)
    preds /= float(len(seeds))
    preds = np.clip(preds, 0.0, ymax)

    # train-only self-consistency of the seed-averaged augmented model
    tr_fit = np.zeros(Fa.shape[0])
    m0 = _mk(1729)
    m0.fit(Fa, y)
    tr_fit = np.clip(m0.predict(Fa), 0.0, ymax)

    findings = {
        'design': 'flat [X|C] rows; ONLY manipulated factor is +25 derived stoichiometry/valence features; learner (HistGB squared_error, 350 iters, lr 0.07, 31 leaves, min_samples_leaf 20, l2 1.0), CV folds (GroupKFold over train element-set groups, 4 folds) and CV seed (1729) identical for both arms; development prediction = mean over 3 seeds of the AUGMENTED model fit on all train rows, clipped to [0, max train y]',
        'engineered_feature_names': ENAMES,
        'cv': {'folds': nsp,
               'group_key': 'train element-set groups',
               'oof_rmse_base': round(_rmse(y, oof['base']), 4),
               'oof_rmse_aug': round(_rmse(y, oof['aug']), 4),
               'oof_delta_aug_minus_base': round(_rmse(y, oof['aug']) - _rmse(y, oof['base']), 4),
               'oof_mae_base': round(float(np.mean(np.abs(y - oof['base']))), 4),
               'oof_mae_aug': round(float(np.mean(np.abs(y - oof['aug']))), 4)},
        'by_chemistry': {'cuprate': block(is_cup), 'fe_based': block(is_fe), 'other': block(is_oth)},
        'valence_feature_coverage': {
            'train_rows_with_cu_valence_defined': int(np.sum(np.isfinite(Etr[:, ENAMES.index('cu_formal_valence')]))),
            'eval_rows_with_cu_valence_defined': int(np.sum(np.isfinite(Ete[:, ENAMES.index('cu_formal_valence')]))),
            'train_cu_hole_p_mean': round(float(np.nanmean(Etr[:, ENAMES.index('cu_hole_p')])), 4),
            'eval_cu_hole_p_mean': round(float(np.nanmean(Ete[:, ENAMES.index('cu_hole_p')])), 4),
            'train_rows_with_fe_valence_defined': int(np.sum(np.isfinite(Etr[:, ENAMES.index('fe_formal_valence')]))),
            'eval_rows_with_fe_valence_defined': int(np.sum(np.isfinite(Ete[:, ENAMES.index('fe_formal_valence')])))},
        'eval_pool': {'rows': int(Ta.shape[0]),
                      'derived_groups': int(len(set(['-'.join([els[j] for j in np.nonzero(Cte[i] > 1e-9)[0]]) for i in range(Cte.shape[0])]))),
                      'pred_mean': round(float(np.mean(preds)), 3),
                      'pred_p90': round(float(np.percentile(preds, 90)), 3),
                      'pred_max': round(float(np.max(preds)), 3),
                      'train_y_mean': round(float(np.mean(y)), 3),
                      'train_y_max': round(ymax, 3)},
        'train_insample_rmse_aug_seed1729': round(_rmse(y, tr_fit), 4),
        'interpretation': 'CALCULATED numbers above are train-only (out-of-fold over element-set groups) plus descriptive statistics of the prediction vector; they are NOT evaluation-label diagnostics, which this code cannot compute. Decision rule stated before execution: a cuprate-localised drop in oof_rmse_aug supports the missing-carrier-count explanation; |oof_delta| < ~0.1 K in every subset supports the alternative that the symmetric descriptors already carry this information and the cuprate error mass is dominated by scatter from unavailable structure/doping/pressure. Only the separately measured development RMSE is out-of-sample evidence, and a train-OOF gain with no development gain would be a transfer counterexample (as with idea-4342ed).',
        'transductive_note': 'evaluation features enter only through the frozen fitted models and through the deterministic per-row engineered transform (a row-wise function of that row composition only); no statistics are pooled across evaluation rows; no evaluation labels, groups, row ids or formulas are used.'}
    return {'prediction': [float(t) for t in preds], 'findings': findings}
