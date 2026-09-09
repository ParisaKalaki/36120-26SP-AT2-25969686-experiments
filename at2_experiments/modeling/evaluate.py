"""Generic model evaluation utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_regression(y_true, y_pred, label: str = "") -> dict:
    """Print and return MAE and RMSE for regression predictions.

    MAE is treated as the primary metric for comparison, since it is
    directly interpretable in the target's own units and is not
    disproportionately influenced by a small number of large errors.
    RMSE is reported as a secondary check for occasional large misses.

    Parameters
    ----------
    y_true, y_pred : array-like
        Actual and predicted target values.
    label : str
        Label printed alongside the metrics (e.g. 'Model - Validation').

    Returns
    -------
    dict
        {'mae': ..., 'rmse': ...}
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label} -- MAE: {mae:.3f} (primary), RMSE: {rmse:.3f}")
    return {"mae": mae, "rmse": rmse}
