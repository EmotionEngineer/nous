from __future__ import annotations
import numpy as np
import torch
from ..facts import PiecewiseLinearCalibratorQuantile

def make_quantile_calibrators(X: np.ndarray, num_bins: int = 8) -> torch.nn.ModuleList:
    """
    Build a list of PiecewiseLinearCalibratorQuantile instances from training data.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data used to compute empirical quantiles per feature.
    num_bins : int, default=8
        Number of bins (i.e., `num_bins + 1` quantile edges).

    Returns
    -------
    calibrators : torch.nn.ModuleList
        One calibrator per feature, ready to pass as `custom_calibrators` to NousNet.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    calibrators = torch.nn.ModuleList()
    for j in range(X.shape[1]):
        col = X[:, j]
        q = np.linspace(0, 1, num_bins + 1)
        edges_np = np.quantile(col, q)
        edges_t = torch.from_numpy(edges_np).float()
        calibrators.append(PiecewiseLinearCalibratorQuantile(edges_t))
    return calibrators