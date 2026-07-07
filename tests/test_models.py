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
