"""Data loading and splitting utilities for the calibration experiment."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ["STAR", "GALAXY", "QSO"]
BANDS = ["u", "g", "r", "i", "z"]
RANDOM_STATE = 42


def features_from_mags(mu, mg, mr, mi, mz):
    """5 dereddened magnitudes + 4 adjacent colors -> (N, 9) feature matrix."""
    return np.column_stack([mu, mg, mr, mi, mz,
                            mu - mg, mg - mr, mr - mi, mi - mz])


def load_real_data(path):
    df = pd.read_csv(path)
    df["class"] = df["class"].astype(str).str.strip().str.upper()
    df = df[df["class"].isin(CLASSES)].copy()
    m = {b: df[f"dered_{b}"].to_numpy(dtype=float) for b in BANDS}
    X = features_from_mags(m["u"], m["g"], m["r"], m["i"], m["z"])
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
    X = features_from_mags(u, g, r, i, z)
    return X, y, r


def make_splits(y, seed):
    """Stratified 60/20/20 (train/calibration/test) split of row indices."""
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    idx_cal, idx_te = train_test_split(idx_tmp, test_size=0.5, stratify=y[idx_tmp],
                                       random_state=seed)
    return idx_tr, idx_cal, idx_te
