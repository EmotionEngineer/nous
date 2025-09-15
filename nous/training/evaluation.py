from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, mean_absolute_error,
    mean_squared_error, r2_score
)

def evaluate_classification(model, test_loader: DataLoader, device, class_names=None):
    """
    Evaluate classification metrics on a dataloader: accuracy and one-vs-rest AUC.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            probs = F.softmax(model(X_batch), dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    all_probs = np.array(all_probs)
    all_preds = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        auc = 0.5
    return acc, auc, all_probs, all_labels

def evaluate_regression(model, test_loader: DataLoader, device, y_scaler=None):
    """
    Evaluate regression: RMSE, MAE, R^2. Returns unscaled predictions if y_scaler is passed.
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy().ravel()
            preds.extend(pred)
            labels.extend(y_batch.numpy().ravel())

    preds = np.array(preds)
    labels = np.array(labels)

    if y_scaler is not None:
        preds_u = y_scaler.inverse_transform(preds.reshape(-1,1)).ravel()
        labels_u = y_scaler.inverse_transform(labels.reshape(-1,1)).ravel()
    else:
        preds_u, labels_u = preds, labels

    rmse = np.sqrt(mean_squared_error(labels_u, preds_u))
    mae  = mean_absolute_error(labels_u, preds_u)
    r2   = r2_score(labels_u, preds_u)
    return rmse, mae, r2, preds_u, labels_u