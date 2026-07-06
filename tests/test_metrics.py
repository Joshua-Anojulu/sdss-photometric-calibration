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
