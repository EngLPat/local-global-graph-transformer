"""
FCLGA GraphTransformer - Path Configuration

Centralized path management for data organization.
Following ML best practices for reproducible research.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories (organized by processing stage)
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_INTERIM = DATA_DIR / "interim"

# Geometry & Simulation directories
GEOMETRY_DIR = DATA_RAW / "geometry"  # .inp files
SIMULATIONS_DIR = DATA_RAW / "simulations"  # .odb files
STRAINS_DIR = DATA_INTERIM / "strains"  # extracted strain .txt files

# Processed datasets
DATASETS_DIR = DATA_PROCESSED / "datasets"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR = RESULTS_DIR / "images"  # mesh visualizations
MODELS_DIR = RESULTS_DIR / "models"  # trained models
BEST_MODELS_DIR = MODELS_DIR / "best"  # best model checkpoints
PLOTS_DIR = RESULTS_DIR / "plots"  # analysis plots
ANIMATIONS_DIR = PLOTS_DIR  # legacy compatibility - plots/animations go here

# Temporary files
TEMP_DIR = PROJECT_ROOT / "temp"

# File paths for key datasets
GEOMETRY_TENSOR = DATA_PROCESSED / "plate_geometry_data.pt"
GEOMETRY_PICKLE = DATA_PROCESSED / "plate_geometry_data.pkl"
NODE_DATA = DATA_PROCESSED / "node_gnn_data.pt"
TRIANGULATION_DATA = DATA_PROCESSED / "triangulation_data.pkl"
STRAINS_TENSOR = DATA_PROCESSED / "strains.pt"


def setup_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        DATA_RAW,
        DATA_PROCESSED,
        DATA_INTERIM,
        GEOMETRY_DIR,
        SIMULATIONS_DIR,
        STRAINS_DIR,
        DATASETS_DIR,
        RESULTS_DIR,
        IMAGES_DIR,
        MODELS_DIR,
        BEST_MODELS_DIR,
        PLOTS_DIR,
        TEMP_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return True


def get_paths_dict():
    """Return dictionary of all paths for easy access."""
    return {
        'project_root': PROJECT_ROOT,
        'data': DATA_DIR,
        'geometry': GEOMETRY_DIR,
        'simulations': SIMULATIONS_DIR,
        'strains': STRAINS_DIR,
        'datasets': DATASETS_DIR,
        'images': IMAGES_DIR,
        'models': MODELS_DIR,
        'plots': PLOTS_DIR,
        'temp': TEMP_DIR,
    }


if __name__ == '__main__':
    # Test: create all directories
    setup_directories()
    print("✓ All directories created successfully!")
    print("\nDirectory structure:")
    for name, path in get_paths_dict().items():
        print(f"  {name:15} -> {path}")
