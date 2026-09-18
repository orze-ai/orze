import json
import numpy as np
from sklearn.linear_model import Ridge


def patient_table(X, y, groups):
    """Patient-level design: per-patient median features, per-patient mean label."""
    g = np.asarray(groups)
    keys = sorted(set(g.tolist()))
    feats = []
    labs = []
    for k in keys:
        m = (g == k)
        feats.append(np.median(X[m], axis=0))
        labs.append(float(np.mean(y[m])))
    return np.asarray(feats, dtype=float), np.asarray(labs, dtype=float), keys


def std_stats(A):
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def fit_ridge(Pf, Lf, alpha):
    mu, sd = std_stats(Pf)
    Z = (Pf - mu) / sd
    shift = float(np.mean(Lf))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Z, Lf - shift)
    return {"mu": mu, "sd": sd, "model": model, "shift": shift}


def ridge_pred(fit, X):
    Z = (X - fit["mu"]) / fit["sd"]
    return np.asarray(fit["model"].predict(Z), dtype=float) + fit["shift"]


def fit_predict(train, inputs, seed=0):
    findings = {
        "method": "row-mean constant shrunk toward patient-level median-feature ridge",
        "selection": "leave-one-training-patient-out row RMSE over (alpha, w) grid",
        "transductive_use": "none: evaluation X used only for the final forward pass",
        "fallback_used": False,
        "fallback_reason": None,
    }
    Xev = np.asarray(inputs["X"], dtype=float)
    n_eval = Xev.shape[0]
    try:
        Xtr = np.asarray(train["X"], dtype=float)
        ytr = np.asarray(train["y"], dtype=float)
        g = np.asarray(list(train["groups"]))
        keys = sorted(set(g.tolist()))
        findings["n_train_rows"] = int(ytr.size)
        findings["n_train_patients"] = int(len(keys))
        findings["n_eval_rows"] = int(n_eval)

        alphas = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
        ws = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

        sse = {}
        sse_med = 0.0
        n_rows = 0
        for k in keys:
            hold = (g == k)
            keep = ~hold
            if int(np.sum(keep)) < 5 or int(np.sum(hold)) < 1:
                continue
            Pf, Lf, _ = patient_table(Xtr[keep], ytr[keep], g[keep])
            base = float(np.mean(ytr[keep]))
            med = float(np.median(ytr[keep]))
            yh = ytr[hold]
            Xh = Xtr[hold]
            n_rows += int(yh.size)
            sse_med += float(np.sum((med - yh) ** 2))
            for a in alphas:
                f = fit_ridge(Pf, Lf, a)
                p = ridge_pred(f, Xh)
                for w in ws:
                    pred = base + w * (p - base)
                    key = (a, w)
                    sse[key] = sse.get(key, 0.0) + float(np.sum((pred - yh) ** 2))

        if n_rows < 1 or not sse:
            raise ValueError("empty LOPO evaluation")

        def rmse(v):
            return float(np.sqrt(v / float(n_rows)))

        items = sorted(sse.items(), key=lambda kv: (kv[1], kv[0][1], kv[0][0]))
        best_alpha, best_w = items[0][0]
        findings["lopo_rows"] = int(n_rows)
        findings["lopo_rmse_constant_mean"] = rmse(sse[(alphas[0], 0.0)])
        findings["lopo_rmse_constant_median"] = rmse(sse_med)
        findings["lopo_rmse_best"] = rmse(items[0][1])
        findings["best_alpha"] = float(best_alpha)
        findings["best_shrinkage_w"] = float(best_w)
        findings["lopo_w_profile_at_best_alpha"] = {
            str(w): rmse(sse[(best_alpha, w)]) for w in ws
        }
        findings["lopo_top5"] = [
            {"alpha": float(a), "w": float(w), "rmse": rmse(v)}
            for (a, w), v in items[:5]
        ]

        Pall, Lall, _ = patient_table(Xtr, ytr, g)
        base_all = float(np.mean(ytr))
        final_fit = fit_ridge(Pall, Lall, best_alpha)
        raw = ridge_pred(final_fit, Xev)
        pred = base_all + float(best_w) * (raw - base_all)
        lo = float(np.min(ytr))
        hi = float(np.max(ytr))
        pred = np.clip(pred, lo, hi)
        if not np.all(np.isfinite(pred)):
            raise ValueError("non-finite predictions")
        findings["constant_base_row_mean"] = base_all
        findings["clip_range"] = [lo, hi]
        findings["pred_mean"] = float(np.mean(pred))
        findings["pred_std"] = float(np.std(pred))
        findings["degenerates_to_constant"] = bool(float(best_w) == 0.0)
        return {"prediction": [float(v) for v in pred], "findings": findings}
    except Exception as exc:
        findings["fallback_used"] = True
        findings["fallback_reason"] = type(exc).__name__ + ": " + str(exc)[:300]
        findings["fallback_rule"] = "constant train row mean (or 0.0 if unavailable)"
        try:
            base = float(np.mean(np.asarray(train["y"], dtype=float)))
        except Exception:
            base = 0.0
            findings["fallback_rule"] = "constant 0.0; train labels unreadable"
        return {"prediction": [base] * n_eval, "findings": findings}
