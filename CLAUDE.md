# CLAUDE.md — orientation for AI coding agents

Research repository for an IEEE URTC paper on **probability calibration** of photometric
star/galaxy/quasar classifiers (SDSS DR17). This file orients you fast; read `REVIEW.md`
before changing anything scientific.

## What this project is

An empirical study, not a library. One script (`src/calibration_experiment.py`) trains four
classifiers on SDSS photometry and measures how well their predicted probabilities are
**calibrated** — overall, across source magnitude, and after post-hoc recalibration. The
paper's claim is *robustness*: capable models stay calibrated across magnitude; the risks are
a weak model (logistic regression) and naive bright-fit recalibration.

## Layout (see README.md for the full tree)

- `src/calibration_experiment.py` — the entire pipeline. No package, no submodules.
- `data/` — `query.sql` (CasJobs), `sdss_sample.csv` (500 rows), and the three metrics CSVs.
- `results/` — the **canonical** metrics CSVs the paper is verified against.
- `figures/` — the four figures in the paper.
- `paper/` — `URTC_paper.docx` (authoritative final) and `URTC_paper.md` (markdown regenerated from it).

## Hard constraints — do not violate

1. **Never commit `sdss.csv`.** The raw ~500k-row catalog (~67 MB) is gitignored by name.
   Only `data/query.sql` and `data/sdss_sample.csv` (500 rows) belong in git.
2. **`--synthetic` output is not science.** `make_synthetic_data()` generates random data for
   code testing only. Never let synthetic numbers or figures reach the paper; the script
   watermarks synthetic figures and prints warnings for this reason.
3. **No leakage across the split.** Models fit on **train** only; recalibration maps fit on
   **calibration** only; all metrics on **test** only. Preserve this separation exactly
   (`main()` in the script). Breaking it invalidates every calibration number.
4. **The paper is authoritative in `paper/URTC_paper.docx`.** `URTC_paper.md` was regenerated
   from that docx and now matches it; keep them in sync if you edit either. If you regenerate
   numbers, reconcile against the docx and `results/`.

## Reproducing

```bash
pip install -r requirements.txt                                   # scikit-learn >= 1.6 required
python src/calibration_experiment.py --data data/sdss.csv --outdir results   # full run (needs the catalog)
python src/calibration_experiment.py --data data/sdss_sample.csv --outdir results_demo  # smoke test
```

The pipeline is single-seed (`RANDOM_STATE = 42`) and deterministic. Re-running on the full
catalog should reproduce `results/` up to library-version numerical drift.

## Conventions

- Classes are ordered `["STAR", "GALAXY", "QSO"]` everywhere; the QSO index is the selection target.
- Magnitude bins have fixed edges `[14,17,18,19,20,22]`; the bright/faint split is `r = 19`.
- ECE uses 15 bins by default, reported for uniform and equal-mass ("quantile") strategies.
- Figures are written by the plotting helpers only; keep filenames `fig1..fig4_*` stable — the
  paper references them by name.

## If asked to publish / commit

Confirm git identity, keep `sdss.csv` gitignored, and do not force-push or rewrite history.
This repo follows the user's standing rule of committing under their own identity (no AI
co-author trailer).
