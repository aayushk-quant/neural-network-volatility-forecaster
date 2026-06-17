import pandas as pd
import statsmodels.api as sm
import numpy as np
from arch import arch_model

HAR_COLS = ['log_rv_1', 'log_rv_5', 'log_rv_22']

def naive_forecast(X: pd.DataFrame, horizon: int) -> pd.Series:
    return X[f'log_rv_{horizon}'].rename(f'naive_{horizon}')

def ewma_forecast(X: pd.DataFrame) -> pd.Series:
    return X['log_ewma'].rename('ewma')

def har_forecast(X_train, y_train, X_test) -> pd.Series:
    design_matrix = sm.add_constant(X_train[HAR_COLS])
    model = sm.OLS(y_train, design_matrix).fit()
    test_design = sm.add_constant(X_test[HAR_COLS], has_constant = 'add')
    preds = model.predict(test_design)
    return pd.Series(preds, index = X_test.index, name = 'har')

def garch_forecast(returns: pd.Series, train_end, horizon: int) -> pd.Series:
    scaled_returns = returns.dropna() * 100
    am = arch_model(scaled_returns, mean='Constant', vol='Garch', p=1, q=1, dist='normal')
    res = am.fit(last_obs=train_end, disp='off')
    fc = res.forecast(horizon=horizon, start=train_end, reindex=False)
    garch_fc = np.log(np.sqrt(fc.variance.mean(axis = 1)) * np.sqrt(252) / 100)
    return garch_fc.rename('garch')