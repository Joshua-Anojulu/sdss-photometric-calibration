# REVIEW.md — code-vs-paper review

Rigorous review of `src/calibration_experiment.py` against the paper
(`paper/URTC_paper.docx`, authoritative) and the committed metrics in `results/`.
Scope: methodology (leakage, ECE/classwise-ECE, recalibration transfer, selection),
number-for-number verification of the paper's tables/text, methodological holes a
reviewer would raise, and repo runnability. **Nothing is fixed here** — this is findings
only, classified **Blocking / Important / Minor**.

Verified on this machine: Python 3.14.5, scikit-learn 1.9.0, numpy 2.4.6, pandas 3.0.3,
scipy 1.17.1, matplotlib 3.10.9.

---

## Summary

The core methodology is sound and every headline number in the final **docx** reproduces
from the CSVs in `results/`. There are **no blocking scientific errors.**

## Status — fixes applied (post-review, before first commit)

The review below is the original assessment; this section records what was fixed. An
independent second-pass code review (superpowers) confirmed the findings and severities and
added #11.

**Fixed:**
- **#1** `paper/URTC_paper.md` regenerated from the corrected docx — no `[__]` placeholders,
  no stale Table II/III references, classwise-ECE section matches the docx and the code.
- **#3** Paper claim "Hyperparameters are selected on validation data" corrected (docx + md)
  to state that fixed hyperparameters are used with no per-model search — matching the code.
- **#4** `plot_ece_vs_magnitude` / `plot_transfer` now guard an empty `by_mag`; the 500-row
  sample runs end-to-end (exit 0).
- **#5** Table I / §III-A LogReg MCE corrected 0.158 → 0.157 (docx + md).
- **#6** §II-D over-claims corrected: "more than one bin count" → "equal-width and equal-mass
  (quantile) bins"; per-class "reliability diagrams" → "calibration errors" (docx + md).
- **#7** Script docstring now lists all outputs (`metrics_selection.csv`, `fig4`).
- **#9 / #11** Data-path convention standardized to `data/sdss.csv` (raw catalog moved there,
  still gitignored) across the script, README, and CLAUDE.md.
- **#10** `requirements.txt` annotated with the exact tested versions for archival reruns.

**Not fixed (deliberately):**
- **#2 No uncertainty quantification** — this is a research change (bootstrap/repeated-split
  CIs, re-running the full 500k pipeline, and re-deriving every number), not a mechanical fix.
  It remains a stated limitation for future work.
- **#8** Temperature-scaling-on-log-probs footnote — left to the author; already documented
  in the code.

Findings below are retained for the record; the CSVs, figures, and `results/` were **not**
regenerated (no numbers changed).

---

## Methodology checks

### Three-way split, no leakage — PASS
`main()` (src/calibration_experiment.py:335–341) splits the row index into
train/calibration/test = **60/20/20, stratified by class** (`train_test_split` with
`test_size=0.4` then `0.5`, `random_state=42`). Because each catalog row is one object,
a row-wise split is an object-wise split — no object appears in two sets.
- Base models are fit on **train only** (`model.fit(Xtr, ytr)`, line 347).
- Every recalibration map is fit **only on the calibration set** (`recalibrate(model,
  Xcal, ycal, Xte, ...)`, lines 374–378) — or on a bright-only subset of it.
- **All** reported metrics (baseline, per-magnitude, selection) are computed on the
  held-out **test set** (`Xte`). No train or calibration data leaks into any reported
  metric. This matches the paper's Section II-B exactly.
- Split sizes reproduce the paper's stated 299,997 / 99,999 / 99,999 (test-set bin counts
  in `metrics_by_magnitude.csv` sum to 99,999 per model). ✔

### ECE and classwise ECE — PASS (implementation correct)
- `ece_score` (line 121): standard top-label ECE = sample-weighted mean |accuracy −
  confidence| over confidence bins; supports uniform and equal-mass (`quantile`) edges;
  correctly includes `conf == 1.0` in the top bin. ✔
- `classwise_ece` (line 138): mean one-vs-rest calibration error across the three classes
  — the correct construction for holding class mix out of the metric, as the paper claims.
  ✔ (Called with `n_bins = 15`, consistent with the top-label ECE.)
- `mce_score`, `brier_multiclass`, `nll_score` are all standard and correct.

### Recalibration transfer — PASS (with one modeling caveat, see Minor)
`recalibrate` (line 225) fits Platt (`sigmoid`), isotonic, or temperature scaling on
`(X_fit, y_fit)` and applies to `X_eval`. Platt/isotonic use
`CalibratedClassifierCV(FrozenEstimator(base_model), ...)` so the base model is **not**
refit — the map is learned only on the calibration data. The "representative" vs
"bright-only" fit is exactly the transfer experiment the paper describes. ✔

### Selection metrics — PASS
QSO selection at `P(QSO) ≥ 0.9` (lines 402–422) computes promised purity (mean predicted
P among selected), achieved purity (true QSO fraction among selected), and completeness
(selected true-QSO / all true-QSO), per magnitude bin and per recalibration condition.
Definitions are correct and match Section II-G. ✔

---

## Number verification (paper → results/)

All Table I values and every in-text statistic were checked against the CSVs.

| Paper claim | Source CSV value | Match |
|---|---|---|
| Table I — LogReg acc 0.809 / ECE 0.076 / Brier 0.312 / NLL 0.555 | 0.8087 / 0.0760 / 0.3121 / 0.5547 | ✔ |
| Table I — RF 0.925 / 0.007 / 0.036 / 0.114 / 0.242 | 0.9248 / 0.00709 / 0.0356 / 0.1137 / 0.2416 | ✔ |
| Table I — HistGB 0.924 / 0.004 / 0.031 / 0.114 / 0.212 | 0.9237 / 0.00352 / 0.0309 / 0.1144 / 0.2118 | ✔ |
| Table I — MLP 0.925 / 0.003 / 0.059 / 0.113 / 0.210 | 0.9245 / 0.00311 / 0.0589 / 0.1133 / 0.2103 | ✔ |
| §III-B RF ECE 0.007 → 0.010 (bright→faint) | ECE_raw 0.00675 → 0.01040 | ✔ |
| §III-B MLP ECE 0.002 → 0.006 | 0.00206 → 0.00592 | ✔ |
| §III-B LogReg ECE 0.17 (bright) → 0.03 (faint) | 0.17274 → 0.03036 | ✔ |
| §III-B (docx) classwise RF 0.005 → 0.008; class terms < 0.013 | cwECE_raw 0.00520 → 0.00760; max class term (galaxy, faint) 0.01245 | ✔ |
| §III-C Platt rep: RF 0.007→≈0.029, MLP 0.003→≈0.031 | ECE_rep_platt RF ≈0.029, MLP ≈0.031–0.035 | ✔ |
| §III-C bright-Platt faint bin: RF 0.061, MLP 0.064 | ECE_bright_platt [20,22) 0.06052 / 0.06424 | ✔ |
| §III-C bright-temperature faint bin: RF 0.016, MLP 0.005 | ECE_bright_temperature [20,22) 0.01639 / 0.00460 | ✔ |
| §III-D RF faint sample 96.3% achieved vs 96.8% promised | raw_achieved 0.96339 / raw_promised 0.96810 | ✔ |
| §III-D RF completeness ≈0.78 [18,19) → ≈0.51 [20,22) | 0.78163 → 0.50522 | ✔ |
| §III-D LogReg faint: promises 92%, delivers 77%, completeness < 0.09 | 0.92335 / 0.76933 / 0.08562 | ✔ |

**One rounding discrepancy (Minor):** Table I lists LogReg **MCE = 0.158**; the CSV value
is **0.157466**, which rounds to **0.157**. Off by 0.001.

---

## Findings

### Blocking
*None.* The committed code runs, the split is leakage-free, and every number in the final
docx reproduces from `results/`.

### Important

1. **`paper/URTC_paper.md` is a pre-final draft that diverges from the authoritative docx
   and from the code.** In Section III-B the markdown still contains unfilled placeholders
   — "galaxy-only ECE is `[__]` in the brightest bin and `[__]` in the faintest" — and
   describes a *top-label ECE restricted to true galaxies* cross-check. The shipped
   `URTC_paper.docx` instead reports the **classwise (one-vs-rest) ECE** that the code
   actually computes (RF 0.005→0.008, per-class terms < 0.013), and it drops the Table II /
   Table III references that the markdown still cites. Anyone treating the `.md` as the
   paper source will see unfilled brackets and a method that does not match the code.
   *Recommendation:* replace `URTC_paper.md` with a markdown export of the final docx (or
   delete it and keep only the docx). The docx itself is correct and needs no change.

2. **No uncertainty quantification.** All results come from a **single** 60/20/20 split at
   one seed (`RANDOM_STATE = 42`); no bootstrap/repeated-split confidence intervals are
   reported. Several central claims rest on small absolute ECE gaps (e.g. 0.003 vs 0.007)
   and on "flat across magnitude" — a reviewer will ask whether these differences and the
   flatness are statistically meaningful. Consider bootstrap CIs on ECE per model/bin.

3. **Paper claims hyperparameter selection the code does not perform.** Section II-C states
   "Hyperparameters are selected on validation data," but `build_models()` (line 213) uses
   fixed hyperparameters and there is no validation search anywhere; the calibration set is
   used only for recalibration. Either implement the selection or soften the sentence.

### Minor

4. **Pipeline crashes on inputs where no magnitude bin reaches ≥50 rows.** When `by_mag` is
   empty (e.g. the 500-row `data/sdss_sample.csv`), `plot_ece_vs_magnitude` calls
   `by_mag.groupby("model")` and raises `KeyError: 'model'` (line 262), after baseline
   metrics and `fig1` are already written. Harmless for the full 500k run, but it means the
   sample cannot be run end-to-end; the README's smoke test therefore uses `--synthetic`.
   A one-line guard (`if by_mag.empty: return`) would fix it. *(Not fixed per review scope.)*

5. **Table I MCE for LogReg is 0.158 but should be 0.157** (rounds down from 0.157466).

6. **"ECE reported for more than one bin count" (§II-D) is only partly done.** The committed
   outputs use a single bin count (15) with two *strategies* (uniform + equal-mass), not
   multiple bin *counts*. Either run a second `--n_bins` value or reword.

7. **"Per-class one-vs-rest reliability diagrams are also reported" (§II-D) are not
   generated.** The code produces only the top-label reliability grid (`fig1`); classwise
   ECE is computed numerically but no per-class reliability *figure* is emitted.

8. **Temperature scaling uses log-probabilities as pseudo-logits** (`fit_temperature`,
   line 190), not true model logits. The code documents this as a deliberate, model-agnostic
   approximation; worth a one-line footnote in the paper for transparency.

9. **Script docstring is stale.** The header (lines 17–23) lists only `fig1`–`fig3` and two
   CSVs as outputs, but the script also writes `metrics_selection.csv` and
   `fig4_selection_quality.png`.

10. **`requirements.txt` is largely unpinned** (only `scikit-learn>=1.6`). Committed results
    were produced under specific library versions; exact reproduction may see minor
    numerical drift. Consider pinning for archival reproducibility.

11. **Docs disagree on the raw catalog's path.** The script's USAGE docstring
    (`src/calibration_experiment.py:11`) says `--data sdss.csv` (repo root, where the real
    ~67 MB file actually sits), while `README.md` and `CLAUDE.md` say `--data data/sdss.csv`.
    Both locations are gitignored so there is no data-safety risk, but pick one convention.
    *(Found on independent second-pass review.)*

---

## Runnability from the README — CONFIRMED (with the sample caveat)

- `pip install -r requirements.txt` covers all imports (numpy, pandas, scipy,
  scikit-learn ≥ 1.6, matplotlib).
- `python src/calibration_experiment.py --synthetic --outdir results_demo` runs the **full**
  pipeline end-to-end (exit 0; all metrics + `fig1`–`fig4`). ✔ Verified.
- `--data data/sdss_sample.csv` does **not** run end-to-end (finding #4). The README is
  written accordingly.
- Full-scale reproduction requires regenerating `sdss.csv` from `data/query.sql` in SDSS
  CasJobs — the raw catalog is intentionally gitignored (~67 MB). The `query.sql` is a
  faithful reconstruction from the paper's Methods (no original `.sql` existed on disk);
  verify it returns the paper's ~499,995 rows before trusting a fresh pull.
