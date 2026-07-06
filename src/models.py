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
