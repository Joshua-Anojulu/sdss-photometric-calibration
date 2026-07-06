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
