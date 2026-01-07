"""
FCLGA GraphTransformer Utilities.

This package contains utility functions for data processing, visualization,
metrics calculation, and optimizer configuration.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

from .data_utils import normalize, unnormalize, get_stats, analyze_node_features
from .optimizer_utils import build_optimizer
from .visualization import plot_results, plot_regression, plot_epochs, load_loss_data
from .metrics import calculate_rmse, calculate_r_squared, calculate_mae, calculate_metrics

__all__ = [
    # Data utilities
    'normalize',
    'unnormalize', 
    'get_stats',
    'analyze_node_features',
    # Optimizer
    'build_optimizer',
    # Visualization
    'plot_results',
    'plot_regression',
    'plot_epochs',
    'load_loss_data',
    # Metrics
    'calculate_rmse',
    'calculate_r_squared',
    'calculate_mae',
    'calculate_metrics',
]
