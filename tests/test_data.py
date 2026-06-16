import numpy as np
import pandas as pd
from src.data import log_returns, trailing_realised_vol, forward_realised_vol

def test_log_returns_length():
    test_frame = pd.DataFrame({'close': [100.0, 110.0, 99.0]})
    r = log_returns(test_frame)
    assert len(r) == 2
    assert abs(r.iloc[0] - np.log(110 / 100)) < 1e-12

def test_no_look_ahead():
    r = pd.Series(0.01 * np.random.default_rng(0).standard_normal(60))
    k = 30
    w = 5
    r2 = r.copy(); r2.iloc[k] += 5.0
    good_r = trailing_realised_vol(r, w)
    bad_r = trailing_realised_vol(r2, w)
    trail_diff = (good_r - bad_r).abs().fillna(0)
    assert (trail_diff.iloc[:k] < 1e-12).all()
    assert (trail_diff > 1e-12).any()
    good_f = forward_realised_vol(r, w)
    bad_f = forward_realised_vol(r2, w)
    fwd_diff = (good_f - bad_f).abs().fillna(0)
    assert (fwd_diff.iloc[k:] < 1e-12).all()