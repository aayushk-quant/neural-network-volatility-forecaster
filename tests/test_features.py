import numpy as np
import pandas as pd
from src.data import build_panel
from src.features import make_features, make_targets, assemble

def _panel(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2010-01-01', periods=n)
    prices = pd.DataFrame({'close': 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))}, index=idx)
    prices.index.name = 'date'
    return build_panel(prices)

def test_shapes():
    panel = _panel()
    X, Y = assemble(panel)
    assert len(X) == len(Y)
    assert Y.shape[1] == 2
    assert X.shape[1] == 7
    assert not X.isna().any().any()
    assert not Y.isna().any().any()
    
def test_no_lookahead():
    panel = _panel()
    k = 100
    panel2 = panel.copy()
    panel2.iloc[k, panel2.columns.get_loc('ret')] += 5.0
    feat_diff = (make_features(panel) - make_features(panel2)).abs().fillna(0).max(axis=1)
    assert (feat_diff.iloc[:k] < 1e-12).all()
    assert (feat_diff > 1e-12).any()
    tgt_diff = (make_targets(panel) - make_targets(panel2)).abs().fillna(0).max(axis=1)
    assert (tgt_diff.iloc[k:] < 1e-12).all()