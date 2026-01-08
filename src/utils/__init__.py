"""
FCLGA GraphTransformer Utilities.

This package contains utility functions for data processing, visualization,
metrics calculation, and optimizer configuration.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

from .data_utils import analyze_node_features, get_stats, normalize, unnormalize
from .metrics import calculate_mae, calculate_metrics, calculate_r_squared, calculate_rmse
from .optimizer_utils import build_optimizer
from .visualization import load_loss_data, plot_epochs, plot_regression, plot_results

__all__ = [
    # Data utilities
    "normalize",
    "unnormalize",
    "get_stats",
    "analyze_node_features",
    # Optimizer
    "build_optimizer",
    # Visualization
    "plot_results",
    "plot_regression",
    "plot_epochs",
    "load_loss_data",
    # Metrics
    "calculate_rmse",
    "calculate_r_squared",
    "calculate_mae",
    "calculate_metrics",
]
