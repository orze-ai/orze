import numpy as np, json, math
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

def analyze(data, history, seed):
    def R(v):
        try:
            v = float(v)
        except Exception:
            return None
        return round(v, 4) if math.isfinite(v) else None
    tr, dv = data['train'], data['development']
    fn = list(tr['feature_names'])
    Xtr = np.asarray(tr['X'], float); ytr = np.asarray(tr['y'], float)
    Xdv = np.asarray(dv['X'], float); ydv = np.asarray(dv['y'], float)
    gtr = np.array(['T' + str(g) for g in tr['groups']]); gdv = np.array(['D' + str(g) for g in dv['groups']])
    low = [f.lower() for f in fn]
    def find(s):
        c = [i for i, f in enumerate(low) if s in f]
        return c[0] if len(c) == 1 else None
    ia, isx, it = find('age'), find('sex'), find('test_time')
    out = {'feature_names': fn, 'roles': {'age': ia, 'sex': isx, 'time': it}, 'notes': []}
    if None in (ia, isx, it):
        out['notes'].append('role lookup failed; positional 0,1,2 used')
        ia, isx, it = 0, 1, 2
    voice = [i for i in range(len(fn)) if i not in (ia, isx, it)]
    Xall = np.vstack([Xtr, Xdv]); yall = np.concatenate([ytr, ydv]); gall = np.concatenate([gtr, gdv])
    L = Xall.copy(); logged = []
    for i in voice:
        if Xall[:, i].min() > 0:
            L[:, i] = np.log(Xall[:, i]); logged.append(fn[i])
    out['log_transformed'] = logged
    bcols = [ia, isx] + voice
    bnames = [fn[i] for i in bcols]
    ntr = len(ytr); N = len(yall)
    def ptable(idx):
        g = gall[idx]; ids = list(dict.fromkeys(g.tolist()))
        pos = [np.where(g == p)[0] for p in ids]
        P = np.array([L[idx[q]][:, bcols].mean(0) for q in pos])
        Y = np.array([yall[idx[q]].mean() for q in pos])
        return ids, pos, P, Y
    def evaluate(trn, te):
        ids, pos, P, Y = ptable(trn)
        tids, tpos, TP, TY = ptable(te)
        med = float(np.median(yall[trn]))
        mu, sd = P.mean(0), P.std(0); sd[sd == 0] = 1.0
        Z = (P - mu) / sd; TZ = (TP - mu) / sd
        preds = {'median': np.full(len(tids), med), 'patient_mean_of_means': np.full(len(tids), float(Y.mean()))}
        for a in (10, 100, 300, 1000, 3000, 10000):
            preds['between_a%g' % a] = Ridge(alpha=a).fit(Z, Y).predict(TZ)
        for a in (300, 3000):
            preds['agesex_a%g' % a] = Ridge(alpha=a).fit(Z[:, :2], Y).predict(TZ[:, :2])
            preds['voice_a%g' % a] = Ridge(alpha=a).fit(Z[:, 2:], Y).predict(TZ[:, 2:])
        d = ((TZ[:, None, :] - Z[None, :, :]) ** 2).sum(2)
        nn = np.argsort(d, 1)[:, :5]
        preds['knn5'] = Y[nn].mean(1)
        rowp = {}
        for name, v in preds.items():
            pv = np.empty(len(te))
            for j, q in enumerate(tpos):
                pv[q] = v[j]
            rowp[name] = pv
        Xr = L[trn]; rm, rs = Xr.mean(0), Xr.std(0); rs[rs == 0] = 1.0
        rowp['row_ridge_a10'] = Ridge(alpha=10).fit((Xr - rm) / rs, yall[trn]).predict((L[te] - rm) / rs)
        rowp['shrink50_between_a300'] = 0.5 * rowp['median'] + 0.5 * rowp['between_a300']
        yt = yall[te]
        stats = {}
        for name, pv in rowp.items():
            e = pv - yt
            stats[name] = [(float((e[q] ** 2).sum()), int(len(q)), float(np.abs(e[q]).mean()), float(e[q].mean())) for q in tpos]
        return tids, stats
    def lopo(idx_all):
        g = gall[idx_all]; ids = list(dict.fromkeys(g.tolist()))
        agg = {}; perpat = {}
        for p in ids:
            tids, stats = evaluate(idx_all[g != p], idx_all[g == p])
            for name, per in stats.items():
                sse, n, mae, se = per[0]
                a = agg.setdefault(name, [0.0, 0, []])
                a[0] += sse; a[1] += n; a[2].append(mae)
                perpat.setdefault(name, {})[p] = [R(mae), R(se)]
        summ = {name: {'row_rmse': R(math.sqrt(a[0] / a[1])), 'eq_patient_mae': R(np.mean(a[2]))} for name, a in agg.items()}
        return summ, perpat
    all_idx = np.arange(N); tr_idx = np.arange(ntr); dv_idx = np.arange(ntr, N)
    s35, pp35 = lopo(all_idx)
    out['lopo35_train_plus_dev'] = s35
    out['lopo35_rank_by_row_rmse'] = sorted(s35, key=lambda k: s35[k]['row_rmse'])[:6]
    s28, _ = lopo(tr_idx)
    out['lopo28_train_only'] = s28
    tids, st = evaluate(tr_idx, dv_idx)
    out['dev_patient_ids'] = tids
    out['train28_to_dev7'] = {name: {'row_rmse': R(math.sqrt(sum(p[0] for p in per) / sum(p[1] for p in per))), 'eq_patient_mae': R(np.mean([p[2] for p in per])), 'per_patient_signed': [R(p[3]) for p in per]} for name, per in st.items()}
    best_between = min([k for k in s35 if k.startswith('between_')], key=lambda k: s35[k]['row_rmse'])
    out['lopo35_best_between'] = best_between
    pm = pp35['median']; pb = pp35[best_between]
    out['lopo35_per_patient_mae_signed'] = {'median': pm, best_between: pb}
    out['lopo35_patients_where_between_beats_median'] = int(sum(1 for p in pm if pb[p][0] is not None and pm[p][0] is not None and pb[p][0] < pm[p][0]))
    ids_d, pos_d, P_d, Y_d = ptable(dv_idx)
    ids_t, pos_t, P_t, Y_t = ptable(tr_idx)
    tmed = float(np.median(ytr))
    out['train_row_median'] = R(tmed)
    out['train_patient_mean_y_quantiles'] = [R(q) for q in np.percentile(Y_t, [0, 25, 50, 75, 100])]
    out['dev_patients'] = [{'id': ids_d[j], 'n': int(len(pos_d[j])), 'mean_y': R(Y_d[j]), 'sd_y': R(ydv[pos_d[j]].std()), 'age': R(Xdv[pos_d[j], ia].mean()), 'sex': R(Xdv[pos_d[j], isx].mean()), 'mean_minus_train_median': R(Y_d[j] - tmed)} for j in range(len(ids_d))]
    within = np.concatenate([ydv[q] - ydv[q].mean() for q in pos_d])
    out['dev_oracle_patient_mean_rmse_within_spread_not_a_floor'] = R(math.sqrt((within ** 2).mean()))
    out['dev_level_only_rmse_of_train_median'] = R(math.sqrt(sum(len(pos_d[j]) * (Y_d[j] - tmed) ** 2 for j in range(len(ids_d))) / len(ydv)))
    ids_a, pos_a, P_a, Y_a = ptable(all_idx)
    corr_b = {}; corr_w = {}
    for j, name in enumerate(bnames):
        if P_a[:, j].std() > 0:
            corr_b[name] = R(spearmanr(P_a[:, j], Y_a)[0])
    yc = np.concatenate([yall[q] - yall[q].mean() for q in pos_a])
    for i in range(len(fn)):
        xc = np.concatenate([L[q, i] - L[q, i].mean() for q in pos_a])
        if xc.std() > 0:
            corr_w[fn[i]] = R(spearmanr(xc, yc)[0])
    out['spearman_between_35_patient_means'] = corr_b
    out['spearman_within_patient_centered_pooled'] = corr_w
    rid2pos = {}
    for i, r in enumerate(dv['row_ids']):
        rid2pos[r] = i
    hist = []
    for h in history:
        if not isinstance(h, dict):
            continue
        pred = h.get('prediction')
        if not isinstance(pred, dict) or not h.get('valid'):
            continue
        try:
            pos = []; vals = []
            for r, v in zip(pred['row_ids'], pred['prediction']):
                if r in rid2pos:
                    pos.append(rid2pos[r]); vals.append(float(v))
            if len(pos) != len(ydv) or len(set(pos)) != len(ydv):
                hist.append({'task_id': h.get('task_id'), 'matched_rows': len(pos)}); continue
            pv = np.empty(len(ydv)); pv[np.array(pos)] = np.array(vals)
            e = pv - ydv
            hist.append({'task_id': h.get('task_id'), 'row_rmse': R(math.sqrt((e ** 2).mean())), 'eq_patient_mae': R(np.mean([np.abs(e[q]).mean() for q in pos_d])), 'per_patient_signed': [R(e[q].mean()) for q in pos_d], 'pred_sd': R(pv.std())})
        except Exception as ex:
            hist.append({'task_id': h.get('task_id'), 'error': type(ex).__name__})
    out['history_on_dev'] = hist
    drop_order = ['lopo35_per_patient_mae_signed', 'spearman_within_patient_centered_pooled', 'lopo28_train_only', 'train28_to_dev7', 'history_on_dev', 'dev_patients']
    s = json.dumps(out)
    for k in drop_order:
        if len(s.encode('utf-8')) <= 8000:
            break
        out.pop(k, None); out['notes'].append('dropped ' + k + ' for size'); s = json.dumps(out)
    return out
