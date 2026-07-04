# Design: Calibration study rigor upgrade (v2) for IEEE URTC submission

**Date:** 2026-07-04
**Repo:** `sdss-photometric-calibration` (branch `rigor-v2`)
**Goal:** Raise the SDSS photometric-calibration study to IEEE URTC submission quality —
verified-reproducible results, seed-averaged point estimates with uncertainty, honest
hyperparameter tuning, closed paper-claim gaps, and modular, tested code with
publication-quality figures.

This is **Approach A** (reproduce + rigor + cheap gap-closures), with **Approach B**
(broader models/methods, calibration-vs-color/redshift) deferred to a later cycle. The paper
is **editable**; reported numbers may change.

---

## Baseline facts (measured 2026-07-04)

- **Determinism confirmed:** re-running the current pipeline at seed 42 on the full
  `data/sdss.csv` reproduced all three committed `results/*.csv` **bit-for-bit identical**,
  including on scikit-learn 1.9. The existing numbers are therefore verified accurate; the
  rigor work is an upgrade, not a correction.
- **Compute:** one full run ≈ **2 minutes**. Budget for the whole effort ≈ 1 hour
  (one grid-search pass + ~20 repeated splits + figure/bootstrap work).

---

## Decisions (locked with the user)

1. **Primary point estimate = seed-averaged mean** over repeated splits (± 95% CI), not the
   single seed-42 value. All tables/figures redrawn at higher quality.
2. **Real hyperparameter tuning** via k-fold CV — the paper's "tuned" claim becomes true.
3. **Light modularization + unit tests** — submission-grade code, easier to extend for B.

---

## Data-use discipline (no leakage — unchanged invariant)

Three disjoint uses, enforced and tested:
- **Train** → fit base models AND select hyperparameters (k-fold CV *within train only*).
- **Calibration** → fit post-hoc recalibration maps only (representative and bright-only).
- **Test** → compute every reported metric only.

Each of the R repeated splits is an independent, class-stratified 60/20/20 partition.

---

## Architecture (light modularization)

Split the single 440-line `src/calibration_experiment.py` into focused modules under `src/`:

| Module | Responsibility | Key public functions |
|---|---|---|
| `data.py` | load/clean SDSS CSV, feature construction, synthetic generator | `load_real_data`, `make_synthetic_data`, `features_from_mags` |
| `metrics.py` | calibration/scoring metrics | `ece_score`, `classwise_ece`, `mce_score`, `brier_multiclass`, `nll_score`, `reliability_curve`, `per_class_reliability` |
| `recalibration.py` | post-hoc maps + temperature scaling | `fit_temperature`, `apply_temperature`, `recalibrate` |
| `models.py` | model zoo + hyperparameter grids + CV selection | `build_models`, `PARAM_GRIDS`, `select_hyperparameters` |
| `experiment.py` | orchestrates one split and the repeated-split protocol; aggregation | `run_one_split`, `run_repeated`, `aggregate` |
| `plots.py` | publication-quality figures (per dataviz guidance) | `plot_reliability_grid`, `plot_per_class_reliability`, `plot_ece_vs_magnitude`, `plot_transfer`, `plot_selection` |
| `cli.py` (or keep `calibration_experiment.py` as thin entry) | argument parsing, wiring | `main` |

Each module is independently testable and small enough to reason about in one read.

---

## Experiment protocol (v2)

1. **Hyperparameter selection (once).** For each model, run `GridSearchCV`
   (k=5, scoring = negative log-loss) over a small documented grid on the **train** split of a
   fixed reference seed. Freeze the selected hyperparameters and record them
   (`results/selected_hyperparameters.json`). Grids kept small (≈3–6 configs/model) to bound
   compute; e.g. RF `n_estimators` × `max_depth`; HistGB `learning_rate` × `max_iter`;
   MLP `hidden_layer_sizes` × `alpha`; LogReg `C`.
2. **Repeated splits.** R = 20 independent stratified 60/20/20 splits (seeds 0–19). For each
   split, using the frozen hyperparameters: fit models on train; compute baseline metrics on
   test; per-magnitude-bin ECE/classwise-ECE for every recalibration condition; and the
   P(QSO) ≥ 0.9 selection metrics.
3. **Aggregation.** For every metric, report the **mean and 95% CI** across the R splits
   (t-interval; also keep the per-split raw values). Persist:
   - `results/raw_splits/split_{seed}_*.csv` (per-split detail),
   - `results/metrics_baseline_agg.csv`, `results/metrics_by_magnitude_agg.csv`,
     `results/metrics_selection_agg.csv` (mean + CI).
4. **ECE binning robustness.** Report ECE at **two bin counts (15 and 10)** plus the
   equal-mass (quantile) variant at 15, so the binning-sensitivity claim is backed.

---

## Figures (publication quality; consult the `dataviz` skill at build time)

- **Fig 1** — top-label reliability grid (as now) **plus a new per-class one-vs-rest
  reliability figure** (`fig1b_perclass_reliability.png`) the paper already claims.
- **Fig 2** — ECE vs magnitude per model, with **95% CI bands** from the repeated splits.
- **Fig 3** — recalibration transfer (bright-fit vs representative), with CI bands.
- **Fig 4** — selection purity/completeness vs magnitude, with CI bands.
- Consistent theme, colorblind-safe palette, legible at IEEE column width, vector-friendly.

---

## Tests (`tests/`)

- `test_metrics.py` — ECE known-value cases (perfect calibration → ≈0; hand-computed small
  example); classwise-ECE sanity; MCE ≥ ECE; Brier/NLL bounds.
- `test_split.py` — train/cal/test index sets are pairwise disjoint and class-stratified.
- `test_smoke.py` — end-to-end on synthetic and on the 500-row sample (must not crash; the
  empty-`by_mag` guard holds).
- Run under `pytest`; wired into `reproduce.sh` as a pre-flight.

---

## Reproducibility infrastructure

- `environment.txt` (or `requirements-lock.txt`) with exact pinned versions.
- `reproduce.sh` (+ optional `Makefile` target): install → run tests → run full repeated-split
  experiment → regenerate `results/` and `figures/`. One command, documented in README.
- README "Reproducibility" section notes the bit-identical seed-42 reproduction and the
  repeated-split protocol.

---

## Paper updates (docx + regenerated md, kept in sync)

- **Table I** → seed-averaged mean ± 95% CI for each metric.
- **Methods** → describe: k-fold CV hyperparameter selection (with the grids), the R=20
  repeated-split protocol, uncertainty reporting (mean ± 95% CI), two ECE bin counts, and the
  per-class reliability figure.
- **Results** → restate headline comparisons ("0.003 vs 0.007", "flat across magnitude",
  selection purity/completeness) with CIs / significance language.
- Add the per-class reliability figure and CI bands to figure captions.
- Update `REVIEW.md`: mark #2 (uncertainty), #3 (tuning), #6 (bin counts / per-class
  diagrams) resolved.

---

## Workflow & safety

- All work on branch **`rigor-v2`** → PR into `main`; `main` stays intact until you approve.
- **Hard rule preserved:** the raw `data/sdss.csv` is never committed (gitignored by name);
  only `query.sql` and the 500-row sample are tracked.
- Commit as the user, no AI co-author trailer. Push is approval-gated.
- Numbers change with seed-averaging → the paper is updated in lockstep; nothing is left
  internally inconsistent.

---

## Out of scope (deferred to Approach B)

Additional model families, additional calibration methods, calibration vs. color/redshift,
and cross-survey transfer. These get their own spec later.

---

## Risks / open items

- **Grid ranges** are proposed, not final; will be listed explicitly in the plan and kept
  small to hold compute near budget.
- **Seed-averaging shifts every reported number**; expected and accepted. The seed-42 values
  remain documented as a verified deterministic reference.
- **R=20** balances CI tightness against ~40 min compute; adjustable if the user wants
  tighter intervals.
