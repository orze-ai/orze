import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

FAM = {
    'alkali': ['Li', 'Na', 'K', 'Rb', 'Cs'],
    'alkaline_earth': ['Be', 'Mg', 'Ca', 'Sr', 'Ba'],
    'rare_earth': ['Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
    'tm_3d': ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'],
    'tm_4d5d': ['Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'],
    'p_metal': ['Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi', 'Sb', 'Ge'],
    'chalcogen': ['O', 'S', 'Se', 'Te'],
    'halogen': ['F', 'Cl', 'Br', 'I'],
    'pnictogen': ['N', 'P', 'As'],
    'light': ['H', 'B', 'C', 'Si'],
}
KEY = ['O', 'Cu', 'Fe', 'As', 'Se', 'Ba', 'Sr', 'Ca', 'Y', 'La', 'Bi', 'Hg', 'Tl', 'Pb', 'Nb', 'B', 'Mg', 'C', 'H']


def _feats(C, elements):
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    idx = {e: i for i, e in enumerate(elements)}

    def frac(e):
        if e in idx:
            return C[:, idx[e]]
        return np.zeros(n)

    cols = []
    fam = {}
    for name, els in FAM.items():
        s = np.zeros(n)
        for e in els:
            s = s + frac(e)
        fam[name] = s
        cols.append(s)
    for e in KEY:
        cols.append(frac(e))
    eps = 1e-6
    O = frac('O')
    Cu = frac('Cu')
    Fe = frac('Fe')
    cols.append(np.clip(O / (Cu + eps), 0.0, 50.0) * (Cu > 0))
    cat = fam['alkaline_earth'] + fam['rare_earth']
    cols.append(np.clip(Cu / (cat + eps), 0.0, 50.0) * (Cu > 0) * (cat > 0))
    cols.append(np.clip(O / (1.0 - O + eps), 0.0, 50.0))
    present = C > 1e-9
    cols.append(present.sum(axis=1).astype(float))
    cols.append(C.max(axis=1))
    P = np.where(present, C, 1.0)
    cols.append(-(C * np.log(P)).sum(axis=1))
    cols.append(((Cu > 0) & (O > 0)).astype(float))
    pn = (frac('As') > 0) | (frac('Se') > 0) | (frac('P') > 0) | (frac('Te') > 0) | (frac('S') > 0)
    cols.append(((Fe > 0) & pn).astype(float))
    cols.append(fam['tm_3d'] + fam['tm_4d5d'] - Cu - Fe)
    cols.append(1.0 - O)
    F = np.column_stack(cols)
    return np.nan_to_num(F, nan=0.0, posinf=50.0, neginf=0.0)


def _design(X, C, elements):
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    F = _feats(C, elements)
    Z = np.hstack([X, C, F])
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train, inputs, seed):
    elements = list(train['elements'])
    Ztr = _design(train['X'], train['C'], elements)
    y = np.asarray(train['y'], dtype=float)
    Zev = _design(inputs['X'], inputs['C'], list(inputs.get('elements', elements)))
    model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=1729)
    model.fit(Ztr, y)
    pred = model.predict(Zev)
    pred = np.nan_to_num(np.asarray(pred, dtype=float), nan=float(np.median(y)))
    pred = np.clip(pred, 0.0, max(float(y.max()) * 1.2, 1.0))
    return [float(v) for v in pred]
