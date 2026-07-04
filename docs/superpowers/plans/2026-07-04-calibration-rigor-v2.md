# Calibration Study Rigor v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the SDSS photometric-calibration study to IEEE URTC submission quality — modular tested code, k-fold-CV hyperparameter tuning, seed-averaged results with 95% CIs over 20 repeated splits, closed paper-claim gaps, and publication-quality figures.

**Architecture:** Refactor the single 440-line `src/calibration_experiment.py` into focused modules (`data`, `metrics`, `recalibration`, `models`, `experiment`, `plots`) plus a thin CLI. Add a `tests/` suite. Replace the single-split protocol with hyperparameter selection (CV on train) followed by R=20 repeated stratified splits, aggregating every metric to mean ± 95% CI. Regenerate `results/` and `figures/`, then update the paper.

**Tech Stack:** Python 3.9+, numpy, pandas, scipy, scikit-learn ≥ 1.6, matplotlib, pytest.

## Global Constraints

- scikit-learn **≥ 1.6** required (`sklearn.frozen.FrozenEstimator`).
- **NEVER commit `data/sdss.csv`** (raw ~67 MB catalog; gitignored by filename `sdss.csv`). Only `data/query.sql` and `data/sdss_sample.csv` are tracked.
- **No leakage:** three disjoint data uses — **train** (fit base models + select hyperparameters via CV), **calibration** (fit recalibration maps only), **test** (compute all reported metrics only).
- Class order is fixed: `CLASSES = ["STAR", "GALAXY", "QSO"]`; QSO is the selection target.
- Magnitude bin edges `[14, 17, 18, 19, 20, 22]`; bright/faint split at `r = 19`.
- Repeated splits: `R = 20`, seeds `0..19`; stratified 60/20/20 by class.
- Commit as the user (Joshua Anojulu); **no AI co-author trailer**. All work on branch `rigor-v2`; push is approval-gated.
- Keep figure filenames stable (`fig1..fig4`), add `fig1b_perclass_reliability.png`; the paper references figures by name.

---

## File Structure

- `src/metrics.py` — calibration/scoring metrics (extracted + new `per_class_reliability`).
- `src/data.py` — data loading, feature construction, synthetic generator, `make_splits`.
- `src/recalibration.py` — Platt/isotonic/temperature maps.
- `src/models.py` — model zoo, hyperparameter grids, CV selection.
- `src/experiment.py` — `run_one_split`, `run_repeated`, `aggregate`, CSV persistence.
- `src/plots.py` — publication figures with CI bands + per-class reliability.
- `src/calibration_experiment.py` — thin CLI entry (arg parsing → experiment → plots).
- `tests/test_metrics.py`, `tests/test_data.py`, `tests/test_recalibration.py`, `tests/test_models.py`, `tests/test_experiment.py`, `tests/test_smoke.py`.
- `reproduce.sh`, `requirements-lock.txt` — reproducibility infra.

The current `src/calibration_experiment.py` is the source of truth for functions being "moved verbatim"; its functions and line anchors are cited per task.

---

### Task 1: Metrics module + tests

**Files:**
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`
- Create: `tests/__init__.py` (empty), `src/__init__.py` (empty)
- Reference: `src/calibration_experiment.py:118-186` (functions to move verbatim)

**Interfaces:**
- Produces: `top_label(probs) -> (conf, pred)`; `ece_score(probs, y, n_bins=15, strategy="uniform") -> float`; `classwise_ece(probs, y, n_bins=15) -> (mean, per_class_list)`; `mce_score(probs, y, n_bins=15) -> float`; `brier_multiclass(probs, y) -> float`; `nll_score(probs, y) -> float`; `reliability_curve(probs, y, n_bins=15) -> (xs, ys)`; `per_class_reliability(probs, y, k, n_bins=10) -> (xs, ys)`.

- [ ] **Step 1: Create empty package markers**

```bash
: > src/__init__.py
mkdir -p tests && : > tests/__init__.py
```

- [ ] **Step 2: Write failing metrics tests**

Create `tests/test_metrics.py`:

```python
import numpy as np
from src import metrics

def test_perfect_calibration_zero_ece():
    # every prediction confident (1.0) and correct -> conf==acc -> ECE 0
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0, 1, 2])
    assert metrics.ece_score(probs, y) == 0.0

def test_known_ece_single_bin():
    # both samples conf 0.8 in one bin; one correct, one wrong -> acc 0.5
    # ECE = |0.5 - 0.8| = 0.3
    probs = np.array([[0.8, 0.2, 0.0], [0.8, 0.2, 0.0]])
    y = np.array([0, 1])
    assert abs(metrics.ece_score(probs, y) - 0.3) < 1e-9

def test_mce_ge_ece():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=500)
    y = rng.integers(0, 3, size=500)
    assert metrics.mce_score(p, y) >= metrics.ece_score(p, y) - 1e-12

def test_classwise_ece_returns_per_class():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([1, 1, 1], size=300)
    y = rng.integers(0, 3, size=300)
    mean, per = metrics.classwise_ece(p, y)
    assert len(per) == 3 and abs(mean - float(np.mean(per))) < 1e-12

def test_per_class_reliability_monotone_bins():
    rng = np.random.default_rng(2)
    p = rng.dirichlet([2, 2, 2], size=400)
    y = rng.integers(0, 3, size=400)
    xs, ys = metrics.per_class_reliability(p, y, k=1, n_bins=10)
    assert len(xs) == len(ys) and np.all((ys >= 0) & (ys <= 1))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.metrics'`).

- [ ] **Step 4: Create `src/metrics.py`**

Move `top_label`, `ece_score`, `classwise_ece`, `mce_score`, `brier_multiclass`, `nll_score`, `reliability_curve` **verbatim** from `src/calibration_experiment.py:118-186` (only add `import numpy as np` at top). Then append the new function:

```python
def per_class_reliability(probs, y, k, n_bins=10):
    """One-vs-rest reliability curve for class k: predicted P(class=k) vs empirical rate."""
    pk = probs[:, k]
    yk = (y == k).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (pk >= lo) & (pk < hi)
        if msk.sum() > 0:
            xs.append(pk[msk].mean())
            ys.append(yk[msk].mean())
    return np.array(xs), np.array(ys)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add src/__init__.py src/metrics.py tests/__init__.py tests/test_metrics.py
git commit -m "Extract metrics module with unit tests and per-class reliability"
```

---

### Task 2: Data module + split helper + tests

**Files:**
- Create: `src/data.py`
- Create: `tests/test_data.py`
- Reference: `src/calibration_experiment.py:71-114` (move `_features_from_mags`, `load_real_data`, `make_synthetic_data` verbatim) and `:335-341` (split logic → generalize into `make_splits`).

**Interfaces:**
- Consumes: nothing.
- Produces: `CLASSES`, `BANDS`, `features_from_mags(...)`, `load_real_data(path) -> (X, y, r)`, `make_synthetic_data(n=60000, seed=42) -> (X, y, r)`, `make_splits(y, seed) -> (idx_tr, idx_cal, idx_te)` (stratified 60/20/20).

- [ ] **Step 1: Write failing data/split tests**

Create `tests/test_data.py`:

```python
import numpy as np
from src import data

def test_splits_disjoint_and_complete():
    _, y, _ = data.make_synthetic_data(n=5000, seed=0)
    tr, cal, te = data.make_splits(y, seed=0)
    s_tr, s_cal, s_te = set(tr), set(cal), set(te)
    assert s_tr.isdisjoint(s_cal) and s_tr.isdisjoint(s_te) and s_cal.isdisjoint(s_te)
    assert len(s_tr | s_cal | s_te) == len(y)

def test_splits_proportions_60_20_20():
    _, y, _ = data.make_synthetic_data(n=10000, seed=1)
    tr, cal, te = data.make_splits(y, seed=1)
    assert abs(len(tr)/len(y) - 0.6) < 0.01
    assert abs(len(cal)/len(y) - 0.2) < 0.01
    assert abs(len(te)/len(y) - 0.2) < 0.01

def test_splits_stratified_class_proportions():
    _, y, _ = data.make_synthetic_data(n=10000, seed=2)
    tr, cal, te = data.make_splits(y, seed=2)
    for idx in (tr, cal, te):
        for c in range(3):
            assert abs((y[idx] == c).mean() - (y == c).mean()) < 0.03
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data.py -q`
Expected: FAIL (`No module named 'src.data'`).

- [ ] **Step 3: Create `src/data.py`**

Add `import numpy as np`, `import pandas as pd`, `from sklearn.model_selection import train_test_split`. Move `_features_from_mags` (rename public `features_from_mags`), `load_real_data`, `make_synthetic_data` verbatim from `src/calibration_experiment.py:71-114`, plus module constants `CLASSES = ["STAR","GALAXY","QSO"]`, `BANDS = ["u","g","r","i","z"]`, `RANDOM_STATE = 42`. Then add:

```python
def make_splits(y, seed):
    """Stratified 60/20/20 (train/calibration/test) split of row indices."""
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    idx_cal, idx_te = train_test_split(idx_tmp, test_size=0.5, stratify=y[idx_tmp],
                                       random_state=seed)
    return idx_tr, idx_cal, idx_te
```

Update `load_real_data`/`make_synthetic_data` to reference `features_from_mags` (the renamed function).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "Extract data module with stratified make_splits helper and tests"
```

---

### Task 3: Recalibration module + test

**Files:**
- Create: `src/recalibration.py`
- Create: `tests/test_recalibration.py`
- Reference: `src/calibration_experiment.py:188-236` (move `fit_temperature`, `apply_temperature`, `recalibrate` verbatim).

**Interfaces:**
- Consumes: `src.metrics.nll_score`.
- Produces: `fit_temperature(probs_calib, y_calib) -> float`; `apply_temperature(probs, T) -> np.ndarray`; `recalibrate(base_model, X_fit, y_fit, X_eval, method) -> np.ndarray` (method ∈ {"none","platt","isotonic","temperature"}).

- [ ] **Step 1: Write failing recalibration test**

Create `tests/test_recalibration.py`:

```python
import numpy as np
from src import recalibration as rc

def test_temperature_preserves_argmax():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=200)
    out = rc.apply_temperature(p, T=2.0)
    assert np.array_equal(p.argmax(1), out.argmax(1))
    assert np.allclose(out.sum(1), 1.0)

def test_fit_temperature_returns_positive_scalar():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([2, 2, 2], size=500)
    y = p.argmax(1)
    T = rc.fit_temperature(p, y)
    assert T > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_recalibration.py -q`
Expected: FAIL (`No module named 'src.recalibration'`).

- [ ] **Step 3: Create `src/recalibration.py`**

Add imports: `import numpy as np`, `from sklearn.calibration import CalibratedClassifierCV`, `from sklearn.frozen import FrozenEstimator`, `from scipy.optimize import minimize_scalar`, `from src.metrics import nll_score`. Move `fit_temperature`, `apply_temperature`, `recalibrate` verbatim from `src/calibration_experiment.py:188-236`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_recalibration.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/recalibration.py tests/test_recalibration.py
git commit -m "Extract recalibration module with tests"
```

---

### Task 4: Models module — grids + CV hyperparameter selection

**Files:**
- Create: `src/models.py`
- Create: `tests/test_models.py`
- Reference: `src/calibration_experiment.py:211-223` (`build_models`).

**Interfaces:**
- Consumes: nothing.
- Produces: `build_models(params=None) -> dict[str, estimator]`; `PARAM_GRIDS -> dict[str, dict]`; `select_hyperparameters(X, y, cv=5, seed=42) -> dict[str, dict]` (best params per model).

- [ ] **Step 1: Write failing models test**

Create `tests/test_models.py`:

```python
import numpy as np
from src import models, data

def test_build_models_default_keys():
    m = models.build_models()
    assert set(m) == {"LogReg", "RandomForest", "HistGB", "MLP"}

def test_select_hyperparameters_returns_grid_members():
    X, y, _ = data.make_synthetic_data(n=2000, seed=0)
    best = models.select_hyperparameters(X, y, cv=3, seed=0)
    assert set(best) == {"LogReg", "RandomForest", "HistGB", "MLP"}
    for name, params in best.items():
        for k, v in params.items():
            assert v in models.PARAM_GRIDS[name][k]

def test_build_models_accepts_selected_params():
    X, y, _ = data.make_synthetic_data(n=1500, seed=1)
    best = models.select_hyperparameters(X, y, cv=3, seed=1)
    built = models.build_models(best)
    built["RandomForest"].fit(X, y)  # must fit without error
    assert set(built) == {"LogReg", "RandomForest", "HistGB", "MLP"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -q`
Expected: FAIL (`No module named 'src.models'`).

- [ ] **Step 3: Create `src/models.py`**

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42

# Small, documented grids (keep compute bounded ~ a few configs/model).
PARAM_GRIDS = {
    "LogReg":       {"C": [0.3, 1.0, 3.0]},
    "RandomForest": {"n_estimators": [200, 300], "max_depth": [None, 20]},
    "HistGB":       {"learning_rate": [0.05, 0.1], "max_iter": [100, 200]},
    "MLP":          {"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [1e-4, 1e-3]},
}

def _base(name, params, seed):
    p = dict(params)
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **p))
    if name == "RandomForest":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **p)
    if name == "HistGB":
        return HistGradientBoostingClassifier(random_state=seed, **p)
    if name == "MLP":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(max_iter=500, random_state=seed, **p))
    raise ValueError(name)

# Defaults preserve the original v1 configuration.
_DEFAULTS = {"LogReg": {"C": 1.0},
             "RandomForest": {"n_estimators": 300, "max_depth": None},
             "HistGB": {"learning_rate": 0.1, "max_iter": 100},
             "MLP": {"hidden_layer_sizes": (64, 32), "alpha": 1e-4}}

def build_models(params=None, seed=RANDOM_STATE):
    params = params or _DEFAULTS
    return {name: _base(name, params.get(name, _DEFAULTS[name]), seed)
            for name in _DEFAULTS}

def _grid_estimator(name, seed):
    # estimator with a `params`-compatible interface for GridSearchCV
    return _base(name, _DEFAULTS[name], seed)

def select_hyperparameters(X, y, cv=5, seed=RANDOM_STATE):
    """Grid-search each model on (X, y) with k-fold CV (neg log-loss). Returns best params.
    For pipeline models the grid keys are prefixed to the final estimator step."""
    best = {}
    step = {"LogReg": "logisticregression", "MLP": "mlpclassifier"}
    for name, grid in PARAM_GRIDS.items():
        est = _grid_estimator(name, seed)
        if name in step:
            search_grid = {f"{step[name]}__{k}": v for k, v in grid.items()}
        else:
            search_grid = grid
        gs = GridSearchCV(est, search_grid, scoring="neg_log_loss", cv=cv, n_jobs=-1)
        gs.fit(X, y)
        best[name] = {k.split("__")[-1]: v for k, v in gs.best_params_.items()}
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "Add models module with grids and k-fold-CV hyperparameter selection"
```

---

### Task 5: Experiment module — one split, repeated splits, aggregation

**Files:**
- Create: `src/experiment.py`
- Create: `tests/test_experiment.py`
- Reference: `src/calibration_experiment.py:344-428` (baseline/by-magnitude/selection logic → `run_one_split`).

**Interfaces:**
- Consumes: `src.data` (`CLASSES`, `make_splits`), `src.models` (`build_models`), `src.recalibration.recalibrate`, `src.metrics` (all).
- Produces:
  - `run_one_split(X, y, r, seed, params, n_bins=15, bright_faint_split=19.0, sel_threshold=0.9) -> dict` with keys `"baseline"`, `"by_magnitude"`, `"selection"` (each a `pandas.DataFrame`).
  - `run_repeated(X, y, r, seeds, params, **kw) -> dict[str, pandas.DataFrame]` (concatenated raw per-split frames, each with a `seed` column).
  - `aggregate(df, group_cols, value_cols) -> pandas.DataFrame` (adds `<col>_mean`, `<col>_lo`, `<col>_hi` 95% CI columns).
  - `BIN_EDGES`, `BIN_LABELS`.

- [ ] **Step 1: Write failing experiment tests**

Create `tests/test_experiment.py`:

```python
import numpy as np, pandas as pd
from src import experiment as ex, data, models

def test_aggregate_mean_and_ci():
    df = pd.DataFrame({"model": ["A"] * 5, "ECE": [0.1, 0.2, 0.3, 0.4, 0.5]})
    out = ex.aggregate(df, ["model"], ["ECE"])
    row = out.iloc[0]
    assert abs(row["ECE_mean"] - 0.3) < 1e-9
    assert row["ECE_lo"] < row["ECE_mean"] < row["ECE_hi"]

def test_run_one_split_shapes():
    X, y, r = data.make_synthetic_data(n=8000, seed=0)
    out = ex.run_one_split(X, y, r, seed=0, params=models._DEFAULTS)
    assert set(out) == {"baseline", "by_magnitude", "selection"}
    assert set(out["baseline"]["model"]) == {"LogReg", "RandomForest", "HistGB", "MLP"}
    assert "ECE" in out["baseline"].columns

def test_run_repeated_tags_seed():
    X, y, r = data.make_synthetic_data(n=6000, seed=1)
    raw = ex.run_repeated(X, y, r, seeds=[0, 1], params=models._DEFAULTS)
    assert set(raw["baseline"]["seed"]) == {0, 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_experiment.py -q`
Expected: FAIL (`No module named 'src.experiment'`).

- [ ] **Step 3: Create `src/experiment.py`**

```python
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

    fitted, baseline_rows = {}, []
    for name, model in build_models(params, seed=seed).items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        p = model.predict_proba(Xte)
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
        cond = {"raw": recalibrate(model, Xcal, ycal, Xte, "none")}
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_experiment.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/experiment.py tests/test_experiment.py
git commit -m "Add experiment module: run_one_split, run_repeated, mean+CI aggregation"
```

---

### Task 6: Plots module — CI bands + per-class reliability

**Files:**
- Create: `src/plots.py`
- Reference: `src/calibration_experiment.py:238-307` (existing plot helpers as a starting point).

**Interfaces:**
- Consumes: `src.experiment.BIN_LABELS`, `src.metrics` (`reliability_curve`, `ece_score`, `per_class_reliability`), aggregated DataFrames from `aggregate`.
- Produces: `plot_reliability_grid(fitted, Xte, yte, outdir, n_bins)`; `plot_per_class_reliability(fitted, Xte, yte, outdir, n_bins=10)`; `plot_ece_vs_magnitude(agg_by_mag, outdir)`; `plot_transfer(agg_by_mag, outdir, feature_model="RandomForest")`; `plot_selection(agg_sel, outdir, sel_threshold, feature_model="RandomForest")`. All figures use CI bands where an aggregated `_lo`/`_hi` exists.

> **At build time, consult the `dataviz` skill** for the palette, band styling, and axis/legend rules before writing this module. Keep filenames stable.

- [ ] **Step 1: Establish matplotlib defaults + first two figures**

Create `src/plots.py`. Start with the `plt.rcParams.update({...})` block moved verbatim from `src/calibration_experiment.py:47-65`, `import os`, `import numpy as np`, `import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt`, and `from src.metrics import reliability_curve, ece_score, per_class_reliability`, `from src.experiment import BIN_LABELS`. Then add:

```python
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
```

- [ ] **Step 2: Add the three CI-band figures**

Append to `src/plots.py`:

```python
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
```

- [ ] **Step 3: Smoke-check the module imports and runs on synthetic aggregates**

Run:
```bash
python -c "from src import plots, experiment as ex, data, models; \
X,y,r=data.make_synthetic_data(n=8000,seed=0); \
raw=ex.run_repeated(X,y,r,seeds=[0,1],params=models._DEFAULTS); \
agg=ex.aggregate(raw['by_magnitude'],['model','mag_bin'],['ECE_raw']); \
import os; os.makedirs('.tmp_fig',exist_ok=True); \
plots.plot_ece_vs_magnitude(agg,'.tmp_fig'); print('ok')"
```
Expected: prints `ok`, writes `.tmp_fig/fig2_ece_vs_magnitude.png`. Then `rm -rf .tmp_fig`.

- [ ] **Step 4: Commit**

```bash
git add src/plots.py
git commit -m "Add plots module with CI bands and per-class reliability figure"
```

---

### Task 7: Thin CLI entry + smoke test

**Files:**
- Rewrite: `src/calibration_experiment.py` (now a thin orchestrator)
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: all modules.
- Produces: CLI `--data`, `--synthetic`, `--outdir`, `--seeds` (int count, default 20), `--n_bins` (default 15), `--sel_threshold` (default 0.9), `--feature_model` (default RandomForest), `--tune/--no-tune` (default tune on real data). Writes aggregated CSVs (`metrics_baseline_agg.csv`, `metrics_by_magnitude_agg.csv`, `metrics_selection_agg.csv`), per-split raw CSVs under `<outdir>/raw_splits/`, `selected_hyperparameters.json`, and all figures.

- [ ] **Step 1: Write failing smoke test**

Create `tests/test_smoke.py`:

```python
import subprocess, sys, os, glob

def test_synthetic_end_to_end(tmp_path):
    out = tmp_path / "res"
    cmd = [sys.executable, "src/calibration_experiment.py", "--synthetic",
           "--seeds", "2", "--no-tune", "--outdir", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(out / "metrics_baseline_agg.csv")
    assert glob.glob(str(out / "fig*_*.png"))
    assert os.path.exists(out / "raw_splits")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smoke.py -q`
Expected: FAIL (old CLI lacks `--seeds/--no-tune` and agg outputs).

- [ ] **Step 3: Rewrite `src/calibration_experiment.py`**

```python
"""Calibration robustness experiment (v2) — thin CLI orchestrator.

  python src/calibration_experiment.py --data data/sdss.csv --outdir results
  python src/calibration_experiment.py --synthetic --seeds 3 --no-tune --outdir results_demo

Real runs use k-fold-CV hyperparameter selection and R repeated stratified splits;
every metric is aggregated to mean + 95% CI. --synthetic uses random data (NOT results).
"""
import argparse, os, json
import numpy as np
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
```

- [ ] **Step 4: Run smoke test to verify it passes**

Run: `python -m pytest tests/test_smoke.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS (all tests green).

- [ ] **Step 6: Commit**

```bash
git add src/calibration_experiment.py tests/test_smoke.py
git commit -m "Rewrite CLI as thin v2 orchestrator (tuning, repeated splits, agg outputs)"
```

---

### Task 8: Reproducibility infra

**Files:**
- Create: `reproduce.sh`
- Create: `requirements-lock.txt`
- Modify: `README.md` (Reproducing section), `requirements.txt` (add pytest)

**Interfaces:** none (scripts/docs).

- [ ] **Step 1: Create `requirements-lock.txt`**

```bash
python - <<'PY'
import importlib.metadata as m
pkgs = ["numpy","pandas","scipy","scikit-learn","matplotlib","pytest"]
with open("requirements-lock.txt","w") as f:
    for p in pkgs:
        try: f.write(f"{p}=={m.version(p)}\n")
        except m.PackageNotFoundError: pass
print(open("requirements-lock.txt").read())
PY
```

- [ ] **Step 2: Create `reproduce.sh`**

```bash
cat > reproduce.sh <<'SH'
#!/usr/bin/env bash
# One-command reproduction of the v2 results and figures.
# Requires data/sdss.csv (regenerate from data/query.sql via SDSS CasJobs; it is gitignored).
set -euo pipefail
pip install -r requirements.txt
python -m pytest -q
if [ ! -f data/sdss.csv ]; then
  echo "ERROR: data/sdss.csv missing. Regenerate it from data/query.sql (SDSS CasJobs)." >&2
  exit 1
fi
python src/calibration_experiment.py --data data/sdss.csv --seeds 20 --outdir results
echo "Done. See results/ and figures/."
SH
chmod +x reproduce.sh
```

- [ ] **Step 3: Add pytest to `requirements.txt` and update README**

Add `pytest` under the functional deps in `requirements.txt`. In `README.md`, replace the "Reproducing the results" run command with the repeated-split workflow and add a note: *"Results reproduce bit-for-bit deterministically at a fixed seed (verified 2026-07-04); the committed numbers are seed-averaged over 20 splits with 95% CIs. Run `./reproduce.sh` for a full rebuild."*

- [ ] **Step 4: Verify reproduce script pre-flight (tests only, no catalog needed)**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reproduce.sh requirements-lock.txt requirements.txt README.md
git commit -m "Add reproducibility infra: reproduce.sh, pinned lock, README workflow"
```

---

### Task 9: Regenerate real results and figures (execution)

**Files:**
- Modify (regenerate): `results/*.csv`, `figures/*.png`, `results/selected_hyperparameters.json`
- Precondition: `data/sdss.csv` present locally.

**Interfaces:** none.

- [ ] **Step 1: Run the full v2 experiment on the real catalog**

Run: `python src/calibration_experiment.py --data data/sdss.csv --seeds 20 --outdir results`
Expected: completes (~40–60 min); prints selected hyperparameters; writes `metrics_*_agg.csv`, `raw_splits/`, `selected_hyperparameters.json`, and `fig1..fig4` + `fig1b`.

- [ ] **Step 2: Sync figures directory**

```bash
cp results/fig*.png figures/
```

- [ ] **Step 3: Sanity-check the aggregates**

Run: `python -c "import pandas as pd; print(pd.read_csv('results/metrics_baseline_agg.csv').round(4).to_string(index=False))"`
Expected: four models with `ECE_mean`/`ECE_lo`/`ECE_hi` columns; RF/HistGB/MLP ECE_mean in ~0.003–0.010, LogReg high.

- [ ] **Step 4: Confirm the raw catalog is still ignored**

Run: `git status --porcelain | grep -E "sdss\.csv$" || echo "OK: sdss.csv not tracked"`
Expected: `OK: sdss.csv not tracked`.

- [ ] **Step 5: Commit regenerated artifacts**

```bash
git add results/ figures/
git commit -m "Regenerate v2 results (seed-averaged, 95% CI) and figures"
```

---

### Task 10: Update the paper and REVIEW.md

**Files:**
- Modify: `paper/URTC_paper.docx` (via python-docx), then regenerate `paper/URTC_paper.md`
- Modify: `REVIEW.md`

**Interfaces:** none.

- [ ] **Step 1: Read the new aggregated numbers**

Run: `python -c "import pandas as pd; [print(f, pd.read_csv(f).round(4).to_string(index=False)) for f in ['results/metrics_baseline_agg.csv','results/metrics_by_magnitude_agg.csv','results/metrics_selection_agg.csv']]"`
Record the seed-averaged mean ± CI values needed for Table I and the Results text.

- [ ] **Step 2: Update Table I and Methods/Results in the docx**

Using python-docx (single-run paragraph edits as established in this repo): set Table I cells to `mean` values and add a "±95% CI" note to the caption; update §II-C to describe k-fold-CV grid search (list the grids); update §II-D to mention two bin counts (15, 10) and the per-class reliability figure; update §II-F/Results to report mean ± 95% CI and add Fig. 1b. Replace headline numbers (0.003 vs 0.007, transfer values, selection purity/completeness) with the new seed-averaged means; keep claims that still hold, soften any that no longer do. Save the docx and re-open with python-docx to verify it is valid.

- [ ] **Step 3: Regenerate the markdown from the corrected docx**

Reuse the docx→markdown conversion approach already used in this repo (headings by pattern, Table I as a markdown table, figure captions, references as an ordered list, inject the two display equations). Write `paper/URTC_paper.md`. Verify: `grep -c "\[__\]" paper/URTC_paper.md` prints `0`.

- [ ] **Step 4: Update REVIEW.md status**

Mark findings #2 (uncertainty quantification), #3 (hyperparameter tuning), and #6 (bin counts / per-class reliability diagrams) as RESOLVED in the "Status" section, noting the v2 protocol (20 splits, mean ± 95% CI, CV tuning, per-class reliability figure). Note that `results/` now holds aggregated CSVs plus `raw_splits/`.

- [ ] **Step 5: Verify docx/md consistency and commit**

Run: `python -c "import docx; d=docx.Document('paper/URTC_paper.docx'); print('ok', len(d.paragraphs))"`
Expected: prints `ok` and a paragraph count.

```bash
git add paper/URTC_paper.docx paper/URTC_paper.md REVIEW.md
git commit -m "Update paper and REVIEW for v2 (seed-averaged results, CIs, tuning, per-class reliability)"
```

---

### Task 11: Finalize branch → PR

**Files:** none (git/gh).

- [ ] **Step 1: Full test + lint pass**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 2: Confirm hard rule before any push**

Run: `git ls-files | grep -E "(^|/)sdss\.csv$" && echo "VIOLATION" || echo "OK: raw catalog not tracked"`
Expected: `OK: raw catalog not tracked`.

- [ ] **Step 3: STOP for user approval to push**

Present a summary of changes and the new headline numbers. **Do not push or open a PR until the user approves** (per repo workflow). On approval: `git push -u origin rigor-v2` and `gh pr create --base main --title "Rigor v2: seed-averaged results, CIs, tuning, tests" --body "<summary>"`. Report the PR URL.

---

## Notes for the implementer

- Run every command from the repo root (`C:\Users\josha\OneDrive\Documents\URTC`) so `from src import ...` resolves; tests rely on the repo root being on `sys.path` (pytest adds it).
- `GridSearchCV` with pipelines needs step-prefixed param keys — handled in `select_hyperparameters`; do not change the `PARAM_GRIDS` keys to include prefixes.
- Keep the empty-`by_mag`/empty-selection guards behavior implicitly (aggregation and plots tolerate empty frames; the reference-split figures use the full catalog where bins are populated).
- Do not delete `data/sdss.csv`; it is the local raw catalog and is gitignored.
