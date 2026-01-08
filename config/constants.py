"""
Visualization and Numerical Constants for FCLGA GraphTransformer.

True constants used throughout the pipeline for visualization, plotting,
and numerical stability. Material properties and geometry parameters that
vary per simulation should be passed as function arguments or read from data.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

# ============================================================================
# Visualization Settings
# ============================================================================

# Plot resolution and format
PLOT_DPI = 300
PLOT_FORMAT = "pdf"  # or 'svg', 'png'

# Colormaps for different visualizations
COLORMAP_STRAIN = "viridis"  # For strain field plots
COLORMAP_ERROR = "Reds"  # For error plots

# Plot transparency settings
PLOT_TRANSPARENT_BACKGROUND = False
PLOT_TEXT_COLOR = "black"

# Figure sizes (inches) - optimized for journal publications
FIGURE_SIZE_SINGLE = (3.5, 3.0)  # Single column
FIGURE_SIZE_DOUBLE = (7.0, 3.0)  # Double column
FIGURE_SIZE_FULL = (5.0, 3.0)  # Full width

# Font settings for LaTeX rendering
USE_LATEX = True
FONT_FAMILY = "serif"
FONT_SERIF = ["CMU Serif", "Computer Modern", "serif"]
FONT_SIZE = 11


# ============================================================================
# Numerical Tolerances
# ============================================================================

# Threshold for filtering padded/zero nodes in regression plots
ZERO_THRESHOLD = 1e-6

# Epsilon for numerical stability in normalization
NORMALIZATION_EPS = 1e-8

# Maximum accumulations for statistics calculation (memory limit)
MAX_ACCUMULATIONS = 10**6


# ============================================================================
# Color Limits for Plots
# ============================================================================

# These can be adjusted based on your specific data ranges
# Format: (min, max) or None for automatic
STRAIN_COLOR_LIMITS = None  # Auto-scale based on data
NOMINAL_ERROR_COLOR_LIMITS = (0.0001, 0.0069)
RELATIVE_ERROR_COLOR_LIMITS = (0, 20)  # Percentage


# ============================================================================
# Model Evaluation Settings
# ============================================================================

# Number of runs for inference time benchmarking
BENCHMARK_NUM_RUNS = 100

# Validation tolerance for missing strain data (as fraction)
VALIDATION_MISSING_DATA_TOLERANCE = 0.05  # 5% missing is acceptable


# ============================================================================
# Data Processing Settings
# ============================================================================

# Node indexing (Abaqus uses 1-based, Python uses 0-based)
ABAQUS_NODE_OFFSET = 1  # Abaqus nodes start from 1

# Batch processing settings
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4


# ============================================================================
# Hexbin Plot Settings (for regression visualization)
# ============================================================================

HEXBIN_GRIDSIZE = 35  # Number of hexagons
HEXBIN_MINCNT = 2  # Minimum count to display hexagon
HEXBIN_VMAX = 50  # Maximum count for colorbar


# ============================================================================
# Export all constants
# ============================================================================

__all__ = [
    # Visualization
    "PLOT_DPI",
    "PLOT_FORMAT",
    "COLORMAP_STRAIN",
    "COLORMAP_ERROR",
    "PLOT_TRANSPARENT_BACKGROUND",
    "PLOT_TEXT_COLOR",
    "FIGURE_SIZE_SINGLE",
    "FIGURE_SIZE_DOUBLE",
    "FIGURE_SIZE_FULL",
    "USE_LATEX",
    "FONT_FAMILY",
    "FONT_SERIF",
    "FONT_SIZE",
    # Numerical
    "ZERO_THRESHOLD",
    "NORMALIZATION_EPS",
    "MAX_ACCUMULATIONS",
    # Color limits
    "STRAIN_COLOR_LIMITS",
    "NOMINAL_ERROR_COLOR_LIMITS",
    "RELATIVE_ERROR_COLOR_LIMITS",
    # Evaluation
    "BENCHMARK_NUM_RUNS",
    "VALIDATION_MISSING_DATA_TOLERANCE",
    # Data processing
    "ABAQUS_NODE_OFFSET",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUM_WORKERS",
    # Hexbin
    "HEXBIN_GRIDSIZE",
    "HEXBIN_MINCNT",
    "HEXBIN_VMAX",
]
