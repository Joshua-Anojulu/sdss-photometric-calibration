# CLAUDE.md — orientation for AI coding agents

Research repository for an IEEE URTC paper on **probability calibration** of photometric
star/galaxy/quasar classifiers (SDSS DR17). This file orients you fast; read `REVIEW.md`
before changing anything scientific.

## What this project is

An empirical study, not a library. The pipeline (four classifiers on SDSS photometry) measures
how well their predicted probabilities are **calibrated** — overall, across source magnitude,
and after post-hoc recalibration. The paper's claim is *robustness*: capable models stay
calibrated across magnitude; the risks are a weak model (logistic regression) and naive
bright-fit recalibration. Results are **seed-averaged over 20 stratified splits with 95% CIs**,
with hyperparameters chosen by k-fold-CV grid search on the training split.

## Layout (see README.md for the full tree)

- `src/` — the pipeline as focused modules: `data.py`, `metrics.py`, `recalibration.py`,
  `models.py` (grids + CV selection), `experiment.py` (one split, repeated splits, aggregation),
  `plots.py`, and `calibration_experiment.py` (thin CLI). `tests/` has the pytest suite.
- `data/` — `query.sql` (CasJobs) and `sdss_sample.csv` (500 rows). The raw `sdss.csv` is gitignored.
- `results/` — the **canonical** aggregated metrics (`metrics_*_agg.csv`, mean + 95% CI) the paper
  is verified against, plus per-seed `raw_splits/` and `selected_hyperparameters.json`.
- `figures/` — the five figures in the paper (`fig1`, `fig1b` per-class, `fig2`–`fig4`).
- `paper/` — `URTC_paper.docx` (authoritative final) and `URTC_paper.md` (markdown regenerated from it).

## Hard constraints — do not violate

1. **Never commit `sdss.csv`.** The raw ~500k-row catalog (~67 MB) is gitignored by name.
   Only `data/query.sql` and `data/sdss_sample.csv` (500 rows) belong in git.
2. **`--synthetic` output is not science.** `make_synthetic_data()` generates random data for
   code testing only. Never let synthetic numbers or figures reach the paper; the script
   watermarks synthetic figures and prints warnings for this reason.
3. **No leakage across the split.** Models fit on **train** only; hyperparameters selected by CV
   on **train** only; recalibration maps fit on **calibration** only; all metrics on **test**
   only (see `experiment.run_one_split`). Breaking this separation invalidates every calibration number.
4. **The paper is authoritative in `paper/URTC_paper.docx`.** `URTC_paper.md` was regenerated
   from that docx and now matches it; keep them in sync if you edit either. If you regenerate
   numbers, reconcile against the docx and `results/`.

## Reproducing

```bash
pip install -r requirements.txt                                          # scikit-learn >= 1.6 required
python src/calibration_experiment.py --data data/sdss.csv --seeds 20 --outdir results  # full run (~6 h; needs the catalog)
python src/calibration_experiment.py --synthetic --seeds 3 --no-tune --outdir results_demo  # smoke test (not science)
./reproduce.sh                                                            # one-command full rebuild (deps, tests, run)
```

Each split is deterministic; the full run does CV tuning once on the seed-0 train split, then 20
repeated stratified splits, and aggregates every metric to mean + 95% CI. A single-seed run
reproduces bit-for-bit. Note: `data/sdss_sample.csv` (500 rows) is too small to run end-to-end
(no magnitude bin fills) — use `--synthetic` for a smoke test.

## Conventions

- Classes are ordered `["STAR", "GALAXY", "QSO"]` everywhere; the QSO index is the selection target.
- Magnitude bins have fixed edges `[14,17,18,19,20,22]`; the bright/faint split is `r = 19`.
- ECE is reported at two bin counts (15 and 10) and for uniform and equal-mass ("quantile") strategies.
- Figures are written by `plots.py` only; keep filenames `fig1`, `fig1b`, `fig2..fig4_*` stable — the
  paper references them by name.

## If asked to publish / commit

Confirm git identity, keep `sdss.csv` gitignored, and do not force-push or rewrite history.
This repo follows the user's standing rule of committing under their own identity (no AI
co-author trailer).
