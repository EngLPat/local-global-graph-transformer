"""
FCLGA GraphTransformer - Path Configuration

Centralized path management for data organization.
Following ML best practices for reproducible research.

Supports both linear (elastic) and nonlinear (plastic) material cases.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def get_paths(material_type="nonlinear"):
    """Get paths for specific material type.
    
    Args:
        material_type: Either "linear" (elastic) or "nonlinear" (plastic)
    
    Returns:
        SimpleNamespace with all path attributes
    """
    from types import SimpleNamespace
    
    # Data directories (organized by processing stage and material type)
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_RAW = DATA_DIR / "raw" / material_type
    DATA_PROCESSED = DATA_DIR / "processed" / material_type
    DATA_INTERIM = DATA_DIR / "interim" / material_type
    
    # Geometry & Simulation directories
    GEOMETRY_DIR = DATA_RAW / "geometry"  # .inp files
    SIMULATIONS_DIR = DATA_RAW / "simulations"  # .odb files
    STRAINS_DIR = DATA_INTERIM / "strains"  # extracted strain .txt files
    
    # Processed datasets
    DATASETS_DIR = DATA_PROCESSED / "datasets"
    
    # Results directories
    RESULTS_DIR = PROJECT_ROOT / "results" / material_type
    IMAGES_DIR = RESULTS_DIR / "images"  # mesh visualizations
    MODELS_DIR = RESULTS_DIR / "models"  # trained models
    BEST_MODELS_DIR = MODELS_DIR / "best"  # best model checkpoints
    TEST_RESULTS_DIR = RESULTS_DIR / "test_results"  # test analysis results
    ANIMATIONS_DIR = TEST_RESULTS_DIR  # legacy compatibility
    
    # Temporary files
    TEMP_DIR = PROJECT_ROOT / "temp" / f"abaqus_scratch_{material_type}"
    
    # File paths for key datasets
    GEOMETRY_TENSOR = DATA_PROCESSED / "plate_geometry_data.pt"
    GEOMETRY_PICKLE = DATA_PROCESSED / "plate_geometry_data.pkl"
    NODE_DATA = DATA_PROCESSED / "node_gnn_data.pt"
    TRIANGULATION_DATA = DATA_PROCESSED / "triangulation_data.pkl"
    STRAINS_TENSOR = DATA_PROCESSED / "strains.pt"
    
    return SimpleNamespace(
        PROJECT_ROOT=PROJECT_ROOT,
        DATA_DIR=DATA_DIR,
        DATA_RAW=DATA_RAW,
        DATA_PROCESSED=DATA_PROCESSED,
        DATA_INTERIM=DATA_INTERIM,
        GEOMETRY_DIR=GEOMETRY_DIR,
        SIMULATIONS_DIR=SIMULATIONS_DIR,
        STRAINS_DIR=STRAINS_DIR,
        DATASETS_DIR=DATASETS_DIR,
        RESULTS_DIR=RESULTS_DIR,
        IMAGES_DIR=IMAGES_DIR,
        MODELS_DIR=MODELS_DIR,
        BEST_MODELS_DIR=BEST_MODELS_DIR,
        TEST_RESULTS_DIR=TEST_RESULTS_DIR,
        ANIMATIONS_DIR=ANIMATIONS_DIR,
        TEMP_DIR=TEMP_DIR,
        GEOMETRY_TENSOR=GEOMETRY_TENSOR,
        GEOMETRY_PICKLE=GEOMETRY_PICKLE,
        NODE_DATA=NODE_DATA,
        TRIANGULATION_DATA=TRIANGULATION_DATA,
        STRAINS_TENSOR=STRAINS_TENSOR,
    )


# Legacy compatibility: default nonlinear paths
_default_paths = get_paths("nonlinear")
DATA_DIR = _default_paths.DATA_DIR
DATA_RAW = _default_paths.DATA_RAW
DATA_PROCESSED = _default_paths.DATA_PROCESSED
DATA_INTERIM = _default_paths.DATA_INTERIM
GEOMETRY_DIR = _default_paths.GEOMETRY_DIR
SIMULATIONS_DIR = _default_paths.SIMULATIONS_DIR
STRAINS_DIR = _default_paths.STRAINS_DIR
DATASETS_DIR = _default_paths.DATASETS_DIR
RESULTS_DIR = _default_paths.RESULTS_DIR
IMAGES_DIR = _default_paths.IMAGES_DIR
MODELS_DIR = _default_paths.MODELS_DIR
BEST_MODELS_DIR = _default_paths.BEST_MODELS_DIR
TEST_RESULTS_DIR = _default_paths.TEST_RESULTS_DIR
ANIMATIONS_DIR = _default_paths.ANIMATIONS_DIR
TEMP_DIR = _default_paths.TEMP_DIR
GEOMETRY_TENSOR = _default_paths.GEOMETRY_TENSOR
GEOMETRY_PICKLE = _default_paths.GEOMETRY_PICKLE
NODE_DATA = _default_paths.NODE_DATA
TRIANGULATION_DATA = _default_paths.TRIANGULATION_DATA
STRAINS_TENSOR = _default_paths.STRAINS_TENSOR


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
        TEST_RESULTS_DIR,  # Changed from PLOTS_DIR, removed BEST_MODELS_DIR
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
