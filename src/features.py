import numpy as np
import pandas as pd
from .data import trailing_realised_vol, forward_realised_vol, TRADING_DAYS

EWMA_LAMBDA = 0.94
HAR_WINDOWS = (1, 5, 22)
LONG_WINDOW = 66

def ewma_vol(returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    var = (returns ** 2).ewm(alpha = (1 - lam), adjust = False).mean()
    return (np.sqrt(var) * np.sqrt(TRADING_DAYS)).rename('ewma_vol')

def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    r = panel['ret']
    feats = pd.DataFrame(index = panel.index)
    for w in HAR_WINDOWS:
        rv = trailing_realised_vol(r, w).clip(lower = 1e-8)
        feats[f'log_rv_{w}'] = np.log(rv)
    rv_long = trailing_realised_vol(r, LONG_WINDOW).clip(lower = 1e-8)
    feats[f'log_rv_{LONG_WINDOW}'] = np.log(rv_long)
    feats['log_ewma'] = np.log(ewma_vol(r).clip(lower = 1e-8))
    feats['ret_last'] = r
    feats['neg_ret_last'] = r.clip(upper = 0.0)
    return feats

def make_targets(panel: pd.DataFrame, horizons = (1, 5)) -> pd.DataFrame:
    r = panel['ret']
    tgt = pd.DataFrame(index = panel.index)
    for h in horizons:
        rv = forward_realised_vol(r, h).clip(lower = 1e-8)
        tgt[f'log_rv_fwd_{h}'] = np.log(rv)
    return tgt

def assemble(panel: pd.DataFrame, horizons = (1, 5)) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = make_features(panel)
    Y = make_targets(panel, horizons)
    full = X.join(Y, how = 'inner').dropna()
    return full[X.columns], full[Y.columns]