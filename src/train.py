import torch
import torch.nn as nn
from .model import VolMLP
import pandas as pd

def train_model(X_train, y_train, n_features, epochs=200, lr=1e-3):
    model = VolMLP(n_features)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = lr)
    model.train()
    for epoch in range(epochs):
        optimiser.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimiser.step()
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

def nn_forecast(X_train, y_train, X_test, epochs = 200, lr = 1e-3) -> pd.Series:
    X_train_s, X_test_s = standardise(X_train, X_test)
    X_train_t = torch.tensor(X_train_s.values(), dtype = torch.float32)
    y_train_t = torch.tensor(y_train.values(), dtype = torch.float32)
    X_test_t = torch.tensor(X_test_s.values(), dtype = torch.float32)
    n_features = X_train.shape[1]
    model = train_model(X_train_t, y_train_t, n_features, epochs = epochs, lr = lr)
    preds = predict(model = model, X_test_tensor = X_test_t)
    return pd.Series(preds, index = X_test.index, name = 'nn')
