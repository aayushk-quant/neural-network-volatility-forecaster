import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252

def download_prices(ticker: str = 'SPY', start: str = '2005-01-01', end: str | None = None) -> pd.DataFrame:
    raw = yf.download(ticker, start = start, end = end, auto_adjust = True, progress = False)
    if raw.empty:
        raise ValueError(f'No data returned for {ticker}')
    close = raw['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = close.to_frame('close')
    out.index.name = 'date'
    return out

def log_returns(price: pd.DataFrame) -> pd.Series:
    r = np.log(price['close']).diff()
    r.name = 'ret'
    return r.dropna()

def trailing_realised_vol(returns: pd.Series, window: int) -> pd.Series:
    rms = np.sqrt((returns ** 2).rolling(window = window, min_periods = window).mean())
    return (rms * np.sqrt(TRADING_DAYS)).rename(f'rv_trailing{window}')

def forward_realised_vol(returns: pd.Series, horizon: int) -> pd.Series:
    rms_fwd = np.sqrt((returns ** 2).rolling(window = horizon, min_periods = horizon).mean().shift(-horizon))
    return (rms_fwd * np.sqrt(TRADING_DAYS)).rename(f'rv_forward{horizon}')

def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    r = log_returns(prices)
    panel = prices.loc[r.index].copy()
    panel['ret'] = r
    return panel