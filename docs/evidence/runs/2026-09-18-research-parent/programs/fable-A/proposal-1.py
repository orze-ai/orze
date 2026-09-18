import json, math
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold


def _rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _patient_mae(err, groups):
    d = {}
    for e, g in zip(err, groups):
        d.setdefault(g, []).append(abs(float(e)))
    return float(np.mean([np.mean(v) for v in d.values()]))


class Const:
    def __init__(self, stat):
        self.stat = stat
    def fit(self, X, y):
        y = np.asarray(y, float)
        self.v = float(np.median(y)) if self.stat == 'median' else float(np.mean(y))
        return self
    def predict(self, X):
        return np.full(len(X), self.v)


class Lin:
    def __init__(self, cols, alpha, log_idx):
        self.cols = list(cols); self.alpha = alpha; self.log_idx = set(log_idx)
    def _f(self, X):
        Z = np.asarray(X, float)[:, self.cols].copy()
        for j, c in enumerate(self.cols):
            if c in self.log_idx:
                Z[:, j] = np.log(np.maximum(Z[:, j], 1e-12))
        return Z
    def fit(self, X, y):
        Z = self._f(X)
        self.sc = StandardScaler().fit(Z)
        self.m = Ridge(alpha=self.alpha).fit(self.sc.transform(Z), np.asarray(y, float))
        return self
    def predict(self, X):
        return self.m.predict(self.sc.transform(self._f(X)))


class Boost:
    def __init__(self, cols, seed):
        self.cols = list(cols); self.seed = seed
    def fit(self, X, y):
        self.m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=100, l2_regularization=10.0, random_state=self.seed)
        self.m.fit(np.asarray(X, float)[:, self.cols], np.asarray(y, float))
        return self
    def predict(self, X):
        return self.m.predict(np.asarray(X, float)[:, self.cols])


def _build(fn, Xtr, seed):
    names = [str(n).lower() for n in fn]
    p = Xtr.shape[1]
    def find(key, default):
        for i, n in enumerate(names):
            if key in n:
                return i
        return default
    age = find('age', 0); sex = find('sex', 1); tt = find('test_time', 2)
    demo = [age, sex]
    voice = [i for i in range(p) if i not in (age, sex, tt)]
    allc = list(range(p))
    log_idx = []
    for c in voice:
        col = Xtr[:, c]
        if np.all(col > 0) and skew(col) > 1.0:
            log_idx.append(c)
    cands = {}
    cands['median'] = Const('median')
    cands['mean'] = Const('mean')
    cands['demo_linear'] = Lin(demo, 1e-3, [])
    cands['demo_time_linear'] = Lin(demo + [tt], 1e-3, [])
    for a in (10.0, 100.0, 1000.0):
        cands['ridge_all_a%d' % int(a)] = Lin(allc, a, [])
        cands['ridge_log_a%d' % int(a)] = Lin(allc, a, log_idx)
    cands['ridge_voice_only_log_a100'] = Lin(voice, 100.0, log_idx)
    cands['hgb_shallow_all'] = Boost(allc, seed)
    cands['hgb_shallow_voice_only'] = Boost(voice, seed)
    return cands, log_idx, {'age': int(age), 'sex': int(sex), 'test_time': int(tt)}


def fit_predict(train, inputs, seed):
    seed = int(seed) if seed is not None else 0
    Xtr = np.asarray(train['X'], float); ytr = np.asarray(train['y'], float)
    Xte = np.asarray(inputs['X'], float)
    med = float(np.median(ytr))
    try:
        groups = np.asarray(train['groups'])
        fn = train['feature_names']
        cands, log_idx, idx = _build(fn, Xtr, seed)
        ng = len(set(groups.tolist()))
        gkf = GroupKFold(n_splits=min(7, ng))
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        g_rmse = {}; g_pmae = {}; r_rmse = {}
        for name, mdl in cands.items():
            pred = np.zeros(len(ytr))
            for tr, te in gkf.split(Xtr, ytr, groups):
                pred[te] = mdl.fit(Xtr[tr], ytr[tr]).predict(Xtr[te])
            g_rmse[name] = round(_rmse(pred, ytr), 4)
            g_pmae[name] = round(_patient_mae(pred - ytr, groups), 4)
            pred2 = np.zeros(len(ytr))
            for tr, te in kf.split(Xtr):
                pred2[te] = mdl.fit(Xtr[tr], ytr[tr]).predict(Xtr[te])
            r_rmse[name] = round(_rmse(pred2, ytr), 4)
        keys = list(cands)
        order = sorted(g_rmse, key=lambda k: (g_rmse[k], keys.index(k)))
        best = order[0]
        model = cands[best].fit(Xtr, ytr)
        p = np.asarray(model.predict(Xte), float)
        lo, hi = float(np.min(ytr)), float(np.max(ytr))
        p = np.clip(p, lo, hi)
        bad = ~np.isfinite(p)
        p[bad] = med
        findings = {'fallback_used': False, 'selected_by_patient_cv_rmse': best, 'ranking': order[:6], 'group_cv_rmse': g_rmse, 'group_cv_patient_mae': g_pmae, 'row_cv_rmse': r_rmse, 'log_columns': [str(fn[c]) for c in log_idx], 'column_roles': idx, 'n_train_rows': int(len(ytr)), 'n_train_patients': int(ng), 'clip_range': [lo, hi], 'n_nonfinite_replaced': int(bad.sum()), 'note': 'selection used only train patients via GroupKFold; evaluation inputs used only for prediction'}
        return {'prediction': [float(v) for v in p], 'findings': findings}
    except Exception as e:
        return {'prediction': [med] * len(Xte), 'findings': {'fallback_used': True, 'fallback': 'train median', 'error': str(e)[:400]}}
