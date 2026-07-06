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
