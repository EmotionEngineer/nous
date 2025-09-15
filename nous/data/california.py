from __future__ import annotations
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

def get_california_housing_data(scale_y: bool = True):
    """
    Load California Housing and return standardized X and (optionally) standardized y.
    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, class_names, task_type, y_scaler
    """
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_scaler = StandardScaler()
    X_train_full = x_scaler.fit_transform(X_train_full)
    X_test = x_scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    y_scaler = None
    if scale_y:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()
        y_val   = y_scaler.transform(y_val.reshape(-1,1)).ravel()
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, None, "regression", y_scaler