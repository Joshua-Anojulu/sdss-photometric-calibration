import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from scipy.optimize import minimize_scalar
from src.metrics import nll_score

def fit_temperature(probs_calib, y_calib):
    """Fit scalar T>0 minimizing NLL, using log-probabilities as pseudo-logits
    (model-agnostic; with true logits, e.g. a torch model, use those directly)."""
    eps = 1e-12
    logp = np.log(np.clip(probs_calib, eps, 1.0))

    def nll_T(T):
        z = logp / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return nll_score(e / e.sum(axis=1, keepdims=True), y_calib)

    return float(minimize_scalar(nll_T, bounds=(0.05, 100.0), method="bounded").x)

def apply_temperature(probs, T, eps=1e-12):
    logp = np.log(np.clip(probs, eps, 1.0))
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def recalibrate(base_model, X_fit, y_fit, X_eval, method):
    """Return class probabilities on X_eval after fitting a post-hoc recalibration map
    on (X_fit, y_fit) for an already-fitted base_model."""
    if method == "none":
        return base_model.predict_proba(X_eval)
    if method == "temperature":
        T = fit_temperature(base_model.predict_proba(X_fit), y_fit)
        return apply_temperature(base_model.predict_proba(X_eval), T)
    sk = {"platt": "sigmoid", "isotonic": "isotonic"}[method]
    cal = CalibratedClassifierCV(FrozenEstimator(base_model), method=sk)
    cal.fit(X_fit, y_fit)
    return cal.predict_proba(X_eval)
