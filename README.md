# Calibration Robustness of Photometric Star/Galaxy/Quasar Classifiers Under Magnitude Shift

Code, data samples, figures, and paper for an IEEE URTC study of probability
**calibration** in photometric star / galaxy / quasar classification on SDSS DR17
(~500k spectroscopically-confirmed sources). The study measures whether four common
classifiers produce trustworthy class probabilities, how that calibration behaves
across source magnitude, and whether post-hoc recalibration fit on bright sources
transfers to faint ones.

**Headline findings.** The random-forest, gradient-boosted, and MLP classifiers are
well calibrated (ECE 0.003–0.007) and stay so across the full magnitude range;
calibration tracks model capability (only logistic regression is badly miscalibrated,
ECE 0.076). Recalibration fit on bright sources transfers poorly to faint ones
(worst for Platt scaling; temperature scaling is safest). For a P(QSO) ≥ 0.9 quasar
sample, purity matches the promised probability at every magnitude; the faint-end cost
falls on **completeness**, not purity.

## Repository layout

```
.
├── src/
│   └── calibration_experiment.py   # end-to-end pipeline (models, metrics, figures)
├── data/
│   ├── query.sql                   # CasJobs SQL to regenerate the raw SDSS catalog
│   ├── sdss_sample.csv             # 500-row representative sample (raw catalog is gitignored)
│   ├── metrics_baseline.csv        # per-model calibration/accuracy on the test set
│   ├── metrics_by_magnitude.csv    # per-model ECE per magnitude bin & recalibration condition
│   └── metrics_selection.csv       # QSO selection purity/completeness by magnitude
├── figures/
│   ├── fig1_reliability_baseline.png
│   ├── fig2_ece_vs_magnitude.png
│   ├── fig3_recalibration_transfer.png
│   └── fig4_selection_quality.png
├── results/                        # canonical metrics CSVs the paper is verified against
│   ├── metrics_baseline.csv
│   ├── metrics_by_magnitude.csv
│   └── metrics_selection.csv
├── paper/
│   ├── URTC_paper.docx             # final paper (authoritative)
│   └── URTC_paper.md               # markdown draft source
├── requirements.txt
├── REVIEW.md                       # code-vs-paper review and methodological notes
├── CLAUDE.md                       # orientation for AI coding agents
└── .gitignore                      # excludes the raw sdss.csv (~67 MB) and build cruft
```

> **The raw catalog `sdss.csv` (~500k rows, ~67 MB) is intentionally NOT in the repo.**
> Regenerate it from `data/query.sql` (see below). `data/sdss_sample.csv` (500 rows) is
> provided so the pipeline can be smoke-tested without the full download.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+ and **scikit-learn ≥ 1.6** (for `sklearn.frozen.FrozenEstimator`).

## Reproducing the results

**1. Regenerate the raw catalog (optional — needed only to reproduce full-scale numbers).**
Run `data/query.sql` in [SDSS CasJobs](https://skyserver.sdss.org/casjobs/) with the
DR17 context (submit as a batch query), export the output table to CSV, and save it as
`data/sdss.csv`.

**2. Run the pipeline on the real catalog:**

```bash
python src/calibration_experiment.py --data data/sdss.csv --outdir results
```

This writes `metrics_baseline.csv`, `metrics_by_magnitude.csv`, `metrics_selection.csv`
and `fig1`–`fig4` into `results/`. The versions already committed here were produced this
way; the paper's tables and text are verified against them (see `REVIEW.md`).

**Smoke test without the full catalog** — run on synthetic data (randomly generated,
enough rows to populate every magnitude bin; **not** scientific results):

```bash
python src/calibration_experiment.py --synthetic --outdir results_demo
```

This exercises the whole pipeline (all metrics + `fig1`–`fig4`) and exits 0.

> **Note:** `data/sdss_sample.csv` is provided to show the exact CSV schema the pipeline
> expects and to check the data-loading + baseline path; it is **too small to run
> end-to-end.** With only 500 rows no magnitude bin reaches the ≥50-row floor, so the
> per-magnitude step produces an empty table and the current code raises `KeyError:
> 'model'` at the first magnitude plot (see `REVIEW.md`, Minor). Use `--synthetic` for a
> full smoke test.

## Method summary

- **Data/features:** five dereddened magnitudes (*u,g,r,i,z*) + four adjacent colors.
  Spectroscopic redshift and *r*-band magnitude are withheld from the features
  (*r* is reserved to define the magnitude shift; *z* would trivialize QSO identity).
- **Three-way split by object, stratified by class (60/20/20):** models are fit on
  **train**, any recalibration map is fit **only** on **calibration**, and every reported
  metric is computed on the held-out **test** set — no leakage.
- **Metrics:** top-label ECE (uniform and equal-mass bins), classwise (one-vs-rest) ECE,
  MCE, multiclass Brier, NLL, and reliability diagrams.
- **Recalibration:** Platt (sigmoid), isotonic, and temperature scaling, each fit on the
  full calibration set and on a bright-only subset to measure transfer to faint bins.
- **Selection experiment:** promised vs. achieved purity and completeness for a
  P(QSO) ≥ 0.9 sample, per magnitude bin.

## Citation

Anojulu, J. *Calibration Robustness of Photometric Star, Galaxy, and Quasar Classifiers
Under Magnitude Shift.* IEEE MIT Undergraduate Research Technology Conference (URTC).
See `paper/URTC_paper.docx`.
