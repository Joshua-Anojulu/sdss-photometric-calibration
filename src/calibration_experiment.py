"""
Calibration Robustness of Photometric Star/Galaxy/Quasar Classifiers Under Magnitude Shift
-------------------------------------------------------------------------------------------
Experiment pipeline for the URTC paper.

USAGE
  # verify the code on synthetic data (CODE TEST ONLY -- not scientific results):
  python calibration_experiment.py --synthetic --outdir results_demo

  # run on your real SDSS CSV (downloaded from CasJobs):
  python calibration_experiment.py --data data/sdss.csv --outdir results

EXPECTED CSV COLUMNS (from the CasJobs query)
  class, dered_u, dered_g, dered_r, dered_i, dered_z
  (magnitude-error columns may be present; they are ignored here.)

OUTPUTS (written to --outdir)
  metrics_baseline.csv        per-model calibration/accuracy on the held-out test set
  metrics_by_magnitude.csv    per-model ECE in each magnitude bin, per recalibration condition
  metrics_selection.csv       per-model QSO selection purity/completeness by magnitude
  fig1_reliability_baseline.png
  fig2_ece_vs_magnitude.png
  fig3_recalibration_transfer.png
  fig4_selection_quality.png

IMPORTANT
  Numbers/figures produced with --synthetic come from randomly generated data and are NOT
  scientific results. Only outputs computed from real SDSS data belong in the paper.
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# publication-quality figure defaults (clean, legible at column width)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

CLASSES = ["STAR", "GALAXY", "QSO"]
BANDS = ["u", "g", "r", "i", "z"]
RANDOM_STATE = 42

# ------------------------------- data -------------------------------

def _features_from_mags(mu, mg, mr, mi, mz):
    """5 dereddened magnitudes + 4 adjacent colors -> (N, 9) feature matrix."""
    return np.column_stack([mu, mg, mr, mi, mz,
                            mu - mg, mg - mr, mr - mi, mi - mz])

def load_real_data(path):
    df = pd.read_csv(path)
    df["class"] = df["class"].astype(str).str.strip().str.upper()
    df = df[df["class"].isin(CLASSES)].copy()
    m = {b: df[f"dered_{b}"].to_numpy(dtype=float) for b in BANDS}
    X = _features_from_mags(m["u"], m["g"], m["r"], m["i"], m["z"])
    y = df["class"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    r = m["r"]
    mags_ok = (X[:, :5] > 5).all(axis=1) & (X[:, :5] < 35).all(axis=1)  # drop sentinel/bad photometry (e.g. -9999)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(r) & mags_ok
    n_drop = int((~ok).sum())
    if n_drop:
        print(f"[load] dropped {n_drop} rows with non-finite or out-of-range magnitudes")
    return X[ok], y[ok], r[ok]

def make_synthetic_data(n=60000, seed=RANDOM_STATE):
    """Synthetic 3-class photometry whose class separability degrades at faint r.
    FOR CODE TESTING ONLY -- not physical, not results."""
    rng = np.random.default_rng(seed)
    priors = np.array([0.35, 0.45, 0.20])           # imbalanced, like SDSS spectroscopy
    y = rng.choice(3, size=n, p=priors)
    r = rng.uniform(15.0, 21.5, size=n)             # r-band magnitude
    centers = {0: np.array([1.2, 0.6, 0.30, 0.15]),  # STAR   (u-g, g-r, r-i, i-z)
               1: np.array([1.6, 1.0, 0.55, 0.35]),  # GALAXY
               2: np.array([0.4, 0.2, 0.15, 0.05])}  # QSO
    sigma = 0.10 + 0.06 * (r - 15.0)                # noisier (less separable) when faint
    colors = np.zeros((n, 4))
    for k in range(3):
        msk = y == k
        colors[msk] = centers[k] + rng.normal(0, 1, size=(msk.sum(), 4)) * sigma[msk, None]
    u_g, g_r, r_i, i_z = colors.T
    g = r + g_r
    u = g + u_g
    i = r - r_i
    z = i - i_z
    X = _features_from_mags(u, g, r, i, z)
    return X, y, r

# ------------------------- calibration metrics -------------------------

def top_label(probs):
    return probs.max(axis=1), probs.argmax(axis=1)

def ece_score(probs, y, n_bins=15, strategy="uniform"):
    conf, pred = top_label(probs)
    correct = (pred == y).astype(float)
    if strategy == "quantile":
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        edges[-1] = 1.0 + 1e-9  # include conf == 1.0 in the top bin
    ece, N = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf >= lo) & (conf < hi)
        if msk.sum() == 0:
            continue
        ece += (msk.sum() / N) * abs(correct[msk].mean() - conf[msk].mean())
    return ece

def classwise_ece(probs, y, n_bins=10):
    """Mean one-vs-rest calibration error over classes (controls for class mix)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9  # include p == 1.0
    n = len(y)
    per_class = []
    for k in range(probs.shape[1]):
        pk, yk = probs[:, k], (y == k).astype(float)
        idx = np.digitize(pk, edges) - 1
        e = 0.0
        for m in range(n_bins):
            mask = idx == m
            if mask.any():
                e += (mask.sum() / n) * abs(yk[mask].mean() - pk[mask].mean())
        per_class.append(float(e))
    return float(np.mean(per_class)), per_class

def mce_score(probs, y, n_bins=15):
    conf, pred = top_label(probs)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    edges[-1] = 1.0 + 1e-9  # include conf == 1.0 in the top bin
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf >= lo) & (conf < hi)
        if msk.sum() > 0:
            gaps.append(abs(correct[msk].mean() - conf[msk].mean()))
    return max(gaps) if gaps else 0.0

def brier_multiclass(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

def nll_score(probs, y, eps=1e-12):
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(p)))

def reliability_curve(probs, y, n_bins=15):
    conf, pred = top_label(probs)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    edges[-1] = 1.0 + 1e-9  # include conf == 1.0 in the top bin
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf >= lo) & (conf < hi)
        if msk.sum() > 0:
            xs.append(conf[msk].mean())
            ys.append(correct[msk].mean())
    return np.array(xs), np.array(ys)

# ------------------------- temperature scaling -------------------------

def fit_temperature(probs_calib, y_calib):
    """Fit scalar T>0 minimizing NLL, using log-probabilities as pseudo-logits
    (model-agnostic; with true logits, e.g. a torch model, use those directly)."""
    eps = 1e-12
    logp = np.log(np.clip(probs_calib, eps, 1.0))

    def nll_T(T):
        z = logp / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return nll_score(e / e.sum(axis=1, keepdims=True), y_calib)

    return float(minimize_scalar(nll_T, bounds=(0.05, 100.0), method="bounded").x)

def apply_temperature(probs, T, eps=1e-12):
    logp = np.log(np.clip(probs, eps, 1.0))
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# ------------------------------- models -------------------------------

def build_models():
    return {
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000)),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                               random_state=RANDOM_STATE),
        "HistGB": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                           random_state=RANDOM_STATE)),
    }

def recalibrate(base_model, X_fit, y_fit, X_eval, method):
    """Return class probabilities on X_eval after fitting a post-hoc recalibration map
    on (X_fit, y_fit) for an already-fitted base_model."""
    if method == "none":
        return base_model.predict_proba(X_eval)
    if method == "temperature":
        T = fit_temperature(base_model.predict_proba(X_fit), y_fit)
        return apply_temperature(base_model.predict_proba(X_eval), T)
    sk = {"platt": "sigmoid", "isotonic": "isotonic"}[method]
    cal = CalibratedClassifierCV(FrozenEstimator(base_model), method=sk)
    cal.fit(X_fit, y_fit)
    return cal.predict_proba(X_eval)

# ------------------------------- plots -------------------------------

def _banner(ax, synthetic):
    if synthetic:
        ax.text(0.5, 0.5, "SYNTHETIC DEMO\nNOT REAL DATA", transform=ax.transAxes,
                fontsize=11, color="red", alpha=0.22, ha="center", va="center", rotation=20)

def plot_reliability_grid(fitted, Xte, yte, outdir, n_bins, synthetic):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (name, model) in zip(axes.ravel(), fitted.items()):
        p = model.predict_proba(Xte)
        xs, ys = reliability_curve(p, yte, n_bins)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.plot(xs, ys, "o-", color="tab:blue")
        ax.set_title(f"{name}  (ECE={ece_score(p, yte, n_bins):.3f})")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        _banner(ax, synthetic)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_reliability_baseline.png"), dpi=130)
    plt.close(fig)

def plot_ece_vs_magnitude(by_mag, outdir, synthetic):
    if by_mag.empty:  # no magnitude bin met the >=50-row floor (e.g. a tiny input sample)
        print("[plot] skipping fig2/fig3: no populated magnitude bins")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, g in by_mag.groupby("model"):
        ax.plot(g["mag_bin"], g["ECE_raw"], "o-", label=name)
    ax.set_xlabel("r-band magnitude bin"); ax.set_ylabel("ECE (uncalibrated)")
    ax.legend(); _banner(ax, synthetic)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_ece_vs_magnitude.png"), dpi=130)
    plt.close(fig)

def plot_transfer(by_mag, outdir, synthetic, feature_model="RandomForest"):
    if by_mag.empty:  # nothing to plot when no magnitude bin was populated
        return
    name = feature_model if (by_mag["model"] == feature_model).any() else by_mag["model"].iloc[0]
    g = by_mag[by_mag["model"] == name]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g["mag_bin"], g["ECE_raw"], "o-", color="black", label="uncalibrated")
    if "ECE_bright_platt" in g:
        ax.plot(g["mag_bin"], g["ECE_bright_platt"], "s--", color="tab:red",
                label="bright-fit Platt")
    if "ECE_bright_temperature" in g:
        ax.plot(g["mag_bin"], g["ECE_bright_temperature"], "^-", color="tab:blue",
                label="bright-fit temperature")
    ax.set_xlabel("r-band magnitude bin"); ax.set_ylabel("ECE")
    ax.legend(); _banner(ax, synthetic)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_recalibration_transfer.png"), dpi=130)
    plt.close(fig)

def plot_selection(selection, sel_threshold, outdir, synthetic, feature_model="RandomForest"):
    """Promised vs achieved purity and completeness by magnitude for the featured model."""
    if selection.empty:
        return
    name = feature_model if (selection["model"] == feature_model).any() else selection["model"].iloc[0]
    g = selection[selection["model"] == name]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(g["mag_bin"], g["raw_promised_purity"], "o--", color="gray",
            label="promised purity (mean P)")
    ax.plot(g["mag_bin"], g["raw_achieved_purity"], "o-", color="tab:blue",
            label="achieved purity")
    ax.plot(g["mag_bin"], g["raw_completeness"], "s-", color="tab:orange",
            label="completeness")
    ax.set_xlabel("r-band magnitude bin")
    ax.set_ylabel(f"fraction (selection at P(QSO) >= {sel_threshold})")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    _banner(ax, synthetic)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_selection_quality.png"), dpi=130)
    plt.close(fig)

# ------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="path to SDSS CSV")
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic data (CODE TEST ONLY)")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--bright_faint_split", type=float, default=19.0)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--sel_threshold", type=float, default=0.9,
                    help="P(QSO) cut for the decision-relevant selection experiment")
    ap.add_argument("--feature_model", type=str, default="RandomForest",
                    help="model featured in fig3/fig4 (transfer and selection)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    synthetic = args.synthetic or args.data is None
    if synthetic:
        print("[WARNING] Using SYNTHETIC data -- outputs are NOT scientific results.")
        X, y, r = make_synthetic_data()
    else:
        X, y, r = load_real_data(args.data)
    print(f"Loaded {len(y)} objects | "
          + ", ".join(f"{c}={int((y == i).sum())}" for i, c in enumerate(CLASSES)))

    # three-way split by object, stratified by class (60/20/20)
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    idx_cal, idx_te = train_test_split(idx_tmp, test_size=0.5, stratify=y[idx_tmp],
                                       random_state=RANDOM_STATE)
    Xtr, ytr = X[idx_tr], y[idx_tr]
    Xcal, ycal, rcal = X[idx_cal], y[idx_cal], r[idx_cal]
    Xte, yte, rte = X[idx_te], y[idx_te], r[idx_te]
    print(f"Split: train={len(idx_tr)}, calib={len(idx_cal)}, test={len(idx_te)}")

    # fit models, baseline calibration on the held-out test set
    fitted, baseline_rows = {}, []
    for name, model in build_models().items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        p = model.predict_proba(Xte)
        _, pred = top_label(p)
        baseline_rows.append({
            "model": name,
            "accuracy": (pred == yte).mean(),
            "ECE": ece_score(p, yte, args.n_bins),
            "ECE_quantile": ece_score(p, yte, args.n_bins, "quantile"),
            "MCE": mce_score(p, yte, args.n_bins),
            "Brier": brier_multiclass(p, yte),
            "NLL": nll_score(p, yte),
        })
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(os.path.join(args.outdir, "metrics_baseline.csv"), index=False)
    print("\n=== Baseline (i.i.d. test set) ===")
    print(baseline.round(4).to_string(index=False))
    plot_reliability_grid(fitted, Xte, yte, args.outdir, args.n_bins, synthetic)

    # magnitude bins + the bright/faint calibration split
    bin_edges = np.array([14, 17, 18, 19, 20, 22], dtype=float)
    bin_labels = [f"[{lo:.0f},{hi:.0f})" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    bright_cal = rcal < args.bright_faint_split

    # ECE per magnitude bin, per recalibration condition (Exp 2 + 3)
    conds, by_mag_rows = {}, []
    for name, model in fitted.items():
        cond = {"raw": recalibrate(model, Xcal, ycal, Xte, "none")}
        for meth in ["platt", "isotonic", "temperature"]:
            cond[f"rep_{meth}"] = recalibrate(model, Xcal, ycal, Xte, meth)
            cond[f"bright_{meth}"] = recalibrate(model, Xcal[bright_cal], ycal[bright_cal],
                                                 Xte, meth)
        conds[name] = cond
        for lo, hi, lab in zip(bin_edges[:-1], bin_edges[1:], bin_labels):
            mb = (rte >= lo) & (rte < hi)
            if mb.sum() < 50:
                continue
            row = {"model": name, "mag_bin": lab, "n": int(mb.sum())}
            for c, P in cond.items():
                row[f"ECE_{c}"] = ece_score(P[mb], yte[mb], args.n_bins)
            # classwise (one-vs-rest) calibration error: the correct way to ask whether
            # calibration, not accuracy, shifts with magnitude once the class mix is held out
            cw_mean, cw_per = classwise_ece(cond["raw"][mb], yte[mb], args.n_bins)
            row["cwECE_raw"] = cw_mean
            for ci, cname in enumerate(CLASSES):
                row[f"cwECE_{cname.lower()}"] = cw_per[ci]
            by_mag_rows.append(row)

    by_mag = pd.DataFrame(by_mag_rows)
    by_mag.to_csv(os.path.join(args.outdir, "metrics_by_magnitude.csv"), index=False)
    print("\n=== ECE by magnitude bin (per model, per condition) ===")
    print(by_mag.round(4).to_string(index=False))
    plot_ece_vs_magnitude(by_mag, args.outdir, synthetic)
    plot_transfer(by_mag, args.outdir, synthetic, args.feature_model)

    # Decision-relevant selection quality for the QSO class (Exp 4)
    QSO = CLASSES.index("QSO")
    sel_rows = []
    for name in fitted:
        cond = conds[name]
        for lo, hi, lab in zip(bin_edges[:-1], bin_edges[1:], bin_labels):
            mb = (rte >= lo) & (rte < hi)
            ytb = yte[mb]
            n_true = int((ytb == QSO).sum())
            if mb.sum() < 50 or n_true < 10:
                continue
            row = {"model": name, "mag_bin": lab, "n_true_qso": n_true}
            for c in ["raw", "rep_temperature", "bright_temperature"]:
                pq = cond[c][mb][:, QSO]
                sel = pq >= args.sel_threshold
                nsel = int(sel.sum())
                row[f"{c}_promised_purity"] = float(pq[sel].mean()) if nsel else np.nan
                row[f"{c}_achieved_purity"] = float((ytb[sel] == QSO).mean()) if nsel else np.nan
                row[f"{c}_completeness"] = float((sel & (ytb == QSO)).sum() / n_true)
                row[f"{c}_n_selected"] = nsel
            sel_rows.append(row)
    selection = pd.DataFrame(sel_rows)
    selection.to_csv(os.path.join(args.outdir, "metrics_selection.csv"), index=False)
    print(f"\n=== QSO selection quality at P(QSO) >= {args.sel_threshold} "
          f"(promised vs achieved purity by magnitude) ===")
    print(selection.round(4).to_string(index=False))
    plot_selection(selection, args.sel_threshold, args.outdir, synthetic, args.feature_model)

    print(f"\nWrote CSVs and figures to '{args.outdir}/'.")
    if synthetic:
        print("[REMINDER] SYNTHETIC outputs -- do NOT put these numbers/figures in the paper.")


if __name__ == "__main__":
    main()
