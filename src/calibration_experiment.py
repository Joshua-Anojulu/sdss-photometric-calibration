"""Calibration robustness experiment (v2) — thin CLI orchestrator.

  python src/calibration_experiment.py --data data/sdss.csv --outdir results
  python src/calibration_experiment.py --synthetic --seeds 3 --no-tune --outdir results_demo

Real runs use k-fold-CV hyperparameter selection and R repeated stratified splits;
every metric is aggregated to mean + 95% CI. --synthetic uses random data (NOT results).
"""
import argparse, os, json, sys
import numpy as np

# allow running this file directly (`python src/calibration_experiment.py ...`), where
# Python only puts this file's own directory on sys.path, not the repo root that "src"
# needs to be importable as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data, models, experiment as ex, plots

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--seeds", type=int, default=20, help="number of repeated splits")
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--sel_threshold", type=float, default=0.9)
    ap.add_argument("--feature_model", type=str, default="RandomForest")
    ap.add_argument("--tune", dest="tune", action="store_true", default=None)
    ap.add_argument("--no-tune", dest="tune", action="store_false")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "raw_splits"), exist_ok=True)

    synthetic = args.synthetic or args.data is None
    if synthetic:
        print("[WARNING] SYNTHETIC data -- outputs are NOT scientific results.")
        X, y, r = data.make_synthetic_data()
        tune = False if args.tune is None else args.tune
    else:
        X, y, r = data.load_real_data(args.data)
        tune = True if args.tune is None else args.tune

    seeds = list(range(args.seeds))
    if tune:
        tr, _, _ = data.make_splits(y, seed=0)
        params = models.select_hyperparameters(X[tr], y[tr], cv=5, seed=0)
        print("[tune] selected:", params)
    else:
        params = models._DEFAULTS
    with open(os.path.join(args.outdir, "selected_hyperparameters.json"), "w") as f:
        json.dump(params, f, indent=2, default=str)

    raw = ex.run_repeated(X, y, r, seeds=seeds, params=params,
                          n_bins=args.n_bins, sel_threshold=args.sel_threshold)
    for k in raw:
        raw[k].to_csv(os.path.join(args.outdir, "raw_splits", f"{k}.csv"), index=False)

    base_vals = ["accuracy", "ECE", "ECE10", "ECE_quantile", "MCE", "Brier", "NLL"]
    agg_base = ex.aggregate(raw["baseline"], ["model"], base_vals)
    mag_vals = [c for c in raw["by_magnitude"].columns
                if c.startswith(("ECE_", "cwECE_"))]
    agg_mag = ex.aggregate(raw["by_magnitude"], ["model", "mag_bin"], mag_vals)
    sel_vals = [c for c in raw["selection"].columns
                if c.endswith(("purity", "completeness"))]
    agg_sel = ex.aggregate(raw["selection"], ["model", "mag_bin"], sel_vals)
    agg_base.to_csv(os.path.join(args.outdir, "metrics_baseline_agg.csv"), index=False)
    agg_mag.to_csv(os.path.join(args.outdir, "metrics_by_magnitude_agg.csv"), index=False)
    agg_sel.to_csv(os.path.join(args.outdir, "metrics_selection_agg.csv"), index=False)

    # figures: reliability from a single reference split (seed 0) for the fitted models
    idx_tr, _, idx_te = data.make_splits(y, seed=0)
    fitted = {n: m.fit(X[idx_tr], y[idx_tr]) for n, m in models.build_models(params, 0).items()}
    plots.plot_reliability_grid(fitted, X[idx_te], y[idx_te], args.outdir, args.n_bins)
    plots.plot_per_class_reliability(fitted, X[idx_te], y[idx_te], args.outdir)
    plots.plot_ece_vs_magnitude(agg_mag, args.outdir)
    plots.plot_transfer(agg_mag, args.outdir, args.feature_model)
    plots.plot_selection(agg_sel, args.outdir, args.sel_threshold, args.feature_model)
    print(f"Wrote aggregated CSVs and figures to '{args.outdir}/'.")

if __name__ == "__main__":
    main()
