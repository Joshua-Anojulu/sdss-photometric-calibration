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
