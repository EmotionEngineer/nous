from __future__ import annotations
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_wine_data():
    """
    Load Wine dataset via ucimlrepo and return standardized splits.
    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, class_names, task_type, y_scaler
    """
    from ucimlrepo import fetch_ucirepo
    wine = fetch_ucirepo(id=109)
    X, y_df = wine.data.features, wine.data.targets
    feature_names = X.columns.tolist()
    y = LabelEncoder().fit_transform(y_df.values.ravel())
    class_names = [f"Class_{i+1}" for i in range(len(np.unique(y)))]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = StandardScaler()
    X_train_full = preprocessor.fit_transform(X_train_full)
    X_test = preprocessor.transform(X_test)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, class_names, "classification", None