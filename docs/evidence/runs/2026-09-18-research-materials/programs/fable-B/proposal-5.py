import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression


def _hgb():
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=1729)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _stats(y, p, mask=None):
    if mask is None:
        mask = np.ones(len(y), bool)
    if mask.sum() == 0:
        return {'n': 0}
    yy = y[mask]
    pp = p[mask]
    return {'n': int(mask.sum()), 'rmse': _rmse(yy, pp), 'mae': float(np.mean(np.abs(yy - pp))), 'bias_y_minus_pred': float(np.mean(yy - pp))}


def fit_predict(train, inputs, seed):
    X = np.asarray(train['X'], float)
    C = np.asarray(train['C'], float)
    y = np.asarray(train['y'], float)
    groups = np.asarray(train['groups'])
    el = list(train['elements'])
    F = np.hstack([X, C])
    Fe = np.hstack([np.asarray(inputs['X'], float), np.asarray(inputs['C'], float)])
    n = len(y)
    if 'Cu' in el and 'O' in el:
        cup = (C[:, el.index('Cu')] > 0) & (C[:, el.index('O')] > 0)
    else:
        cup = np.zeros(n, bool)
    fold = np.zeros(n, int)
    oof = np.zeros(n)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(F, y, groups)):
        m = _hgb().fit(F[tr], y[tr])
        oof[te] = m.predict(F[te])
        fold[te] = k
    raw = np.clip(oof, 0.0, None)
    iso_oof = np.zeros(n)
    lin_oof = np.zeros(n)
    slopes = []
    intercepts = []
    for k in range(5):
        tr = fold != k
        te = fold == k
        iso = IsotonicRegression(y_min=0.0, out_of_bounds='clip').fit(raw[tr], y[tr])
        iso_oof[te] = iso.predict(raw[te])
        A = np.vstack([raw[tr], np.ones(int(tr.sum()))]).T
        coef = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        slopes.append(float(coef[0]))
        intercepts.append(float(coef[1]))
        lin_oof[te] = np.clip(coef[0] * raw[te] + coef[1], 0.0, None)
    variants = {'raw_clipped': raw, 'isotonic_nested': iso_oof, 'linear_nested': lin_oof}
    f = {'n_train_rows': int(n), 'n_train_cuprate_rows': int(cup.sum()), 'linear_slopes_per_fold': slopes, 'linear_intercepts_per_fold': intercepts, 'train_groupkfold5_by_element_set': {}}
    for name, p in variants.items():
        f['train_groupkfold5_by_element_set'][name] = {
            'all': _stats(y, p),
            'cuprate': _stats(y, p, cup),
            'noncuprate': _stats(y, p, ~cup),
            'y_gt40': _stats(y, p, y > 40),
            'y_lt10': _stats(y, p, y < 10),
            'pred_gt40': _stats(y, p, p > 40),
            'pred_lt10': _stats(y, p, p < 10),
        }
    qs = np.quantile(raw, np.linspace(0, 1, 11))
    curve = []
    for i in range(10):
        if i == 9:
            m = (raw >= qs[i]) & (raw <= qs[i + 1])
        else:
            m = (raw >= qs[i]) & (raw < qs[i + 1])
        if m.sum() > 0:
            curve.append([float(np.mean(raw[m])), float(np.mean(y[m])), int(m.sum())])
    f['raw_oof_decile_curve_mean_pred_mean_y_count'] = curve
    tail = {}
    for lo, hi in [(40, 60), (60, 80), (80, 100), (100, 1000)]:
        m = (raw >= lo) & (raw < hi)
        if m.sum() > 0:
            tail['%d_%d' % (lo, hi)] = [float(np.mean(raw[m])), float(np.mean(y[m])), int(m.sum())]
    f['raw_oof_high_pred_bins_mean_pred_mean_y_count'] = tail
    final = _hgb().fit(F, y)
    pe_raw = np.clip(final.predict(Fe), 0.0, None)
    iso_full = IsotonicRegression(y_min=0.0, out_of_bounds='clip').fit(raw, y)
    pe = iso_full.predict(pe_raw)
    pe = np.where(np.isfinite(pe), pe, pe_raw)
    f['deployed'] = 'HGB(250 iters, lr 0.08, 31 leaves, l2=1, seed 1729) on X+C, predictions clipped at 0, then isotonic map fitted on train GroupKFold OOF predictions vs train labels; applied to evaluation predictions'
    f['n_eval_rows'] = int(len(pe))
    f['eval_raw_mean'] = float(np.mean(pe_raw))
    f['eval_cal_mean'] = float(np.mean(pe))
    f['eval_raw_frac_gt40'] = float(np.mean(pe_raw > 40))
    f['eval_cal_frac_gt40'] = float(np.mean(pe > 40))
    f['eval_mean_abs_shift'] = float(np.mean(np.abs(pe - pe_raw)))
    f['eval_max_abs_shift'] = float(np.max(np.abs(pe - pe_raw)))
    f['note'] = 'All CV numbers are train-only (GroupKFold by element set), not development scores. Calibrators are scored with a second-level fold rotation over OOF predictions (nested). The seed argument is ignored; HGB uses fixed seed 1729 to match earlier methods. The isotonic map is a learned transformation applied to evaluation-pool predictions; it uses no evaluation labels. Deployed variant is always the isotonic-calibrated one regardless of CV outcome.'
    return {'prediction': [float(v) for v in pe], 'findings': f}
