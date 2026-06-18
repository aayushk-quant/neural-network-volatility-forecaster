import torch
import torch.nn as nn
from .model import VolMLP
import pandas as pd

def train_model(X_train, y_train, X_val, y_val, n_features, max_epochs=200, lr=1e-3, weight_decay = 1e-4, patience = 25):
    model = VolMLP(n_features)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    best_val, best_state, wait = float('inf'), None, 0
    for _ in range(max_epochs):
        model.train()
        optimiser.zero_grad()
        preds = model(X_train)
        loss_fn(preds, y_train).backward()
        optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def standardise(X_train: pd.DataFrame, X_test: pd.DataFrame):
    mu = X_train.mean()
    sigma = X_train.std()
    return ((X_train - mu) / sigma), ((X_test - mu) / sigma)

def predict(model, X_test_tensor):
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
    return preds.numpy()

def nn_forecast(X_train, y_train, X_test) -> pd.Series:
    torch.manual_seed(42)
    n_val = int(len(X_train) * 0.2)
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]
    X_tr_s, X_val_s = standardise(X_tr, X_val)
    _, X_test_s     = standardise(X_tr, X_test)
    mu_y, sd_y = y_tr.mean(), y_tr.std()
    y_tr_s  = (y_tr  - mu_y) / sd_y
    y_val_s = (y_val - mu_y) / sd_y
    X_tr_st = torch.tensor(X_tr_s.values, dtype = torch.float32)
    X_val_st = torch.tensor(X_val_s.values, dtype = torch.float32)
    X_test_st = torch.tensor(X_test_s.values, dtype = torch.float32)
    y_tr_st = torch.tensor(y_tr_s.values, dtype = torch.float32)
    y_val_st = torch.tensor(y_val_s.values, dtype = torch.float32)
    n_features = X_train.shape[1]
    model = train_model(X_tr_st, y_tr_st, X_val_st, y_val_st, n_features)
    preds = predict(model = model, X_test_tensor = X_test_st) * sd_y + mu_y
    return pd.Series(preds, index = X_test.index, name = 'nn')
