import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import reliability_curve, ece_score, per_class_reliability
from src.experiment import BIN_LABELS

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


def plot_reliability_grid(fitted, Xte, yte, outdir, n_bins):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (name, model) in zip(axes.ravel(), fitted.items()):
        p = model.predict_proba(Xte)
        xs, ys = reliability_curve(p, yte, n_bins)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.plot(xs, ys, "o-", color="tab:blue")
        ax.set_title(f"{name}  (ECE={ece_score(p, yte, n_bins):.3f})")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_reliability_baseline.png"))
    plt.close(fig)

def plot_per_class_reliability(fitted, Xte, yte, outdir, n_bins=10):
    classes = ["STAR", "GALAXY", "QSO"]
    fig, axes = plt.subplots(len(fitted), 3, figsize=(9, 3 * len(fitted)))
    for i, (name, model) in enumerate(fitted.items()):
        p = model.predict_proba(Xte)
        for k in range(3):
            ax = axes[i, k]
            xs, ys = per_class_reliability(p, yte, k, n_bins)
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            ax.plot(xs, ys, "o-", color="tab:green")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if i == 0: ax.set_title(classes[k])
            if k == 0: ax.set_ylabel(f"{name}\nempirical")
            if i == len(fitted) - 1: ax.set_xlabel(f"P({classes[k]})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1b_perclass_reliability.png"))
    plt.close(fig)


def _band(ax, g, ycol, color, label, marker="o-"):
    # capture the drawn line's color so a passed color=None (default cycle) still bands correctly
    line, = ax.plot(g["mag_bin"], g[f"{ycol}_mean"], marker, color=color, label=label)
    if f"{ycol}_lo" in g:
        ax.fill_between(range(len(g)), g[f"{ycol}_lo"], g[f"{ycol}_hi"],
                        color=line.get_color(), alpha=0.18, linewidth=0)

def plot_ece_vs_magnitude(agg_by_mag, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, g in agg_by_mag.groupby("model", sort=False):
        g = g.reset_index(drop=True)
        _band(ax, g, "ECE_raw", None, name)
    ax.set_xlabel("r-band magnitude bin"); ax.set_ylabel("ECE (uncalibrated)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_ece_vs_magnitude.png"))
    plt.close(fig)

def plot_transfer(agg_by_mag, outdir, feature_model="RandomForest"):
    g = agg_by_mag[agg_by_mag["model"] == feature_model].reset_index(drop=True)
    if g.empty: return
    fig, ax = plt.subplots(figsize=(7, 5))
    _band(ax, g, "ECE_raw", "black", "uncalibrated")
    _band(ax, g, "ECE_bright_platt", "tab:red", "bright-fit Platt", "s--")
    _band(ax, g, "ECE_bright_temperature", "tab:blue", "bright-fit temperature", "^-")
    ax.set_xlabel("r-band magnitude bin"); ax.set_ylabel("ECE"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_recalibration_transfer.png"))
    plt.close(fig)

def plot_selection(agg_sel, outdir, sel_threshold, feature_model="RandomForest"):
    g = agg_sel[agg_sel["model"] == feature_model].reset_index(drop=True)
    if g.empty: return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    _band(ax, g, "raw_promised_purity", "gray", "promised purity", "o--")
    _band(ax, g, "raw_achieved_purity", "tab:blue", "achieved purity")
    _band(ax, g, "raw_completeness", "tab:orange", "completeness", "s-")
    ax.set_xlabel("r-band magnitude bin")
    ax.set_ylabel(f"fraction (selection at P(QSO) >= {sel_threshold})")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig4_selection_quality.png"))
    plt.close(fig)
