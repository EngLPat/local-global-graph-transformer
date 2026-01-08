"""
Metrics and evaluation utilities for FCLGA GraphTransformer.

Functions for calculating performance metrics like RMSE, R², and other
evaluation statistics.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import numpy as np


def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Square Error.

    Parameters
    ----------
    actual : np.ndarray
        Ground truth values
    predicted : np.ndarray
        Predicted values

    Returns
    -------
    float
        RMSE value
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_r_squared(actual, predicted):
    """
    Calculate R-squared (coefficient of determination).

    Parameters
    ----------
    actual : np.ndarray
        Ground truth values
    predicted : np.ndarray
        Predicted values

    Returns
    -------
    float
        R² value
    """
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot)


def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error.

    Parameters
    ----------
    actual : np.ndarray
        Ground truth values
    predicted : np.ndarray
        Predicted values

    Returns
    -------
    float
        MAE value
    """
    return np.mean(np.abs(actual - predicted))


def calculate_metrics(actual, predicted, filter_zeros=True, threshold=1e-6):
    """
    Calculate comprehensive metrics for regression evaluation.

    Parameters
    ----------
    actual : np.ndarray
        Ground truth values
    predicted : np.ndarray
        Predicted values
    filter_zeros : bool, optional
        Whether to filter out near-zero values (default: True)
    threshold : float, optional
        Threshold for filtering (default: 1e-6)

    Returns
    -------
    dict
        Dictionary containing RMSE, R², MAE, and filtered data info
    """
    if filter_zeros:
        non_zero_mask = np.abs(actual) > threshold
        filtered_actual = actual[non_zero_mask]
        filtered_predicted = predicted[non_zero_mask]
    else:
        filtered_actual = actual
        filtered_predicted = predicted

    if len(filtered_actual) == 0:
        return {
            "rmse": np.nan,
            "r_squared": np.nan,
            "mae": np.nan,
            "num_points": 0,
            "filtered_points": 0,
        }

    return {
        "rmse": calculate_rmse(filtered_actual, filtered_predicted),
        "r_squared": calculate_r_squared(filtered_actual, filtered_predicted),
        "mae": calculate_mae(filtered_actual, filtered_predicted),
        "num_points": len(actual),
        "filtered_points": len(filtered_actual),
    }


__all__ = [
    "calculate_rmse",
    "calculate_r_squared",
    "calculate_mae",
    "calculate_metrics",
]
