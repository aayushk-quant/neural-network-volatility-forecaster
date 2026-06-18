import numpy as np
import pandas as pd
from scipy import stats
from .baselines import naive_forecast, ewma_forecast, har_forecast, garch_forecast
from .train import nn_forecast

def rmse(forecast, truth):
    return np.sqrt(np.mean((forecast - truth) ** 2))

def qlike_loss(forecast, truth):
    var_f = np.exp(2 * forecast)
    var_t = np.exp(2 * truth)
    ratio = var_t / var_f
    return ratio - np.log(ratio) - 1

def qlike(forecast, truth):
    return np.mean(qlike_loss(forecast, truth))

def diebold_mariano(loss_a, loss_b, horizon):
    d = np.asarray(loss_a) - np.asarray(loss_b)
    T = len(d)
    d_bar = d.mean()
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv = gamma0
    for k in range(1, horizon):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        lrv += 2 * gamma_k
    dm = d_bar / np.sqrt(lrv / T)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return dm, p

def walk_forward(X, Y, returns, horizon, initial_train = 750, step = 252) -> pd.DataFrame:
    target_col = f'log_rv_fwd_{horizon}'
    preds = {'naive': [], 'ewma': [], 'har': [], 'nn': [], 'garch': []}
    truths = []
    for start in range(initial_train, len(X), step):
        X_train = X.iloc[:start]
        y_train = Y[target_col].iloc[:start]
        X_test = X.iloc[start:start + step]
        if len(X_test) == 0:
            break
        preds['naive'].append(naive_forecast(X_test, horizon))
        preds['ewma'].append(ewma_forecast(X_test))
        preds['har'].append(har_forecast(X_train, y_train, X_test))
        preds['nn'].append(nn_forecast(X_train, y_train, X_test))
        train_end = X.index[start - 1]
        g = garch_forecast(returns, train_end, horizon)
        preds['garch'].append(g.reindex(X_test.index))
        truths.append(Y[target_col].iloc[start:start + step])
    out = {m: pd.concat(parts) for m, parts in preds.items()}
    out['truth'] = pd.concat(truths)
    return pd.DataFrame(out)