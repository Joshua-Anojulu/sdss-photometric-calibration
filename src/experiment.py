import numpy as np
import pandas as pd
from scipy import stats
from src.data import CLASSES, make_splits
from src.models import build_models
from src.recalibration import recalibrate
from src.metrics import (top_label, ece_score, classwise_ece, mce_score,
                         brier_multiclass, nll_score)

BIN_EDGES = np.array([14, 17, 18, 19, 20, 22], dtype=float)
BIN_LABELS = [f"[{lo:.0f},{hi:.0f})" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
QSO = CLASSES.index("QSO")

def run_one_split(X, y, r, seed, params, n_bins=15, bright_faint_split=19.0,
                  sel_threshold=0.9):
    idx_tr, idx_cal, idx_te = make_splits(y, seed)
    Xtr, ytr = X[idx_tr], y[idx_tr]
    Xcal, ycal, rcal = X[idx_cal], y[idx_cal], r[idx_cal]
    Xte, yte, rte = X[idx_te], y[idx_te], r[idx_te]

    fitted, probs_te, baseline_rows = {}, {}, []
    for name, model in build_models(params, seed=seed).items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        p = model.predict_proba(Xte)
        probs_te[name] = p  # reused as the uncalibrated ("raw") condition below
        _, pred = top_label(p)
        baseline_rows.append({
            "model": name, "accuracy": (pred == yte).mean(),
            "ECE": ece_score(p, yte, n_bins),
            "ECE10": ece_score(p, yte, 10),
            "ECE_quantile": ece_score(p, yte, n_bins, "quantile"),
            "MCE": mce_score(p, yte, n_bins),
            "Brier": brier_multiclass(p, yte), "NLL": nll_score(p, yte),
        })
    baseline = pd.DataFrame(baseline_rows)

    bright_cal = rcal < bright_faint_split
    conds, by_mag_rows = {}, []
    for name, model in fitted.items():
        cond = {"raw": probs_te[name]}
        for meth in ["platt", "isotonic", "temperature"]:
            cond[f"rep_{meth}"] = recalibrate(model, Xcal, ycal, Xte, meth)
            cond[f"bright_{meth}"] = recalibrate(model, Xcal[bright_cal],
                                                 ycal[bright_cal], Xte, meth)
        conds[name] = cond
        for lo, hi, lab in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
            mb = (rte >= lo) & (rte < hi)
            if mb.sum() < 50:
                continue
            row = {"model": name, "mag_bin": lab, "n": int(mb.sum())}
            for c, P in cond.items():
                row[f"ECE_{c}"] = ece_score(P[mb], yte[mb], n_bins)
            cw_mean, cw_per = classwise_ece(cond["raw"][mb], yte[mb], n_bins)
            row["cwECE_raw"] = cw_mean
            for ci, cname in enumerate(CLASSES):
                row[f"cwECE_{cname.lower()}"] = cw_per[ci]
            by_mag_rows.append(row)
    by_mag = pd.DataFrame(by_mag_rows)

    sel_rows = []
    for name in fitted:
        cond = conds[name]
        for lo, hi, lab in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
            mb = (rte >= lo) & (rte < hi)
            ytb = yte[mb]
            n_true = int((ytb == QSO).sum())
            if mb.sum() < 50 or n_true < 10:
                continue
            row = {"model": name, "mag_bin": lab, "n_true_qso": n_true}
            for c in ["raw", "rep_temperature", "bright_temperature"]:
                pq = cond[c][mb][:, QSO]
                sel = pq >= sel_threshold
                nsel = int(sel.sum())
                row[f"{c}_promised_purity"] = float(pq[sel].mean()) if nsel else np.nan
                row[f"{c}_achieved_purity"] = float((ytb[sel] == QSO).mean()) if nsel else np.nan
                row[f"{c}_completeness"] = float((sel & (ytb == QSO)).sum() / n_true)
                row[f"{c}_n_selected"] = nsel
            sel_rows.append(row)
    selection = pd.DataFrame(sel_rows)
    return {"baseline": baseline, "by_magnitude": by_mag, "selection": selection}

def run_repeated(X, y, r, seeds, params, **kw):
    acc = {"baseline": [], "by_magnitude": [], "selection": []}
    for s in seeds:
        out = run_one_split(X, y, r, seed=int(s), params=params, **kw)
        for k, df in out.items():
            df = df.copy(); df["seed"] = int(s); acc[k].append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}

def aggregate(df, group_cols, value_cols):
    """Mean + 95% CI (t-interval across seeds) for each value column."""
    def _ci(g):
        out = {}
        for c in value_cols:
            x = g[c].dropna().to_numpy()
            m = float(np.mean(x)) if len(x) else np.nan
            if len(x) > 1:
                se = stats.sem(x); h = se * stats.t.ppf(0.975, len(x) - 1)
            else:
                h = 0.0
            out[f"{c}_mean"] = m; out[f"{c}_lo"] = m - h; out[f"{c}_hi"] = m + h
        return pd.Series(out)
    return df.groupby(group_cols, sort=False).apply(_ci, include_groups=False).reset_index()
