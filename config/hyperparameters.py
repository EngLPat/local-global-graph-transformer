"""
FCLGA GraphTransformer - Hyperparameters Configuration

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London

Centralized configuration for all hyperparameters and settings.
Users can modify these values to adapt the pipeline to their needs.
"""

import torch


class ModelConfig:
    """Model architecture hyperparameters"""
    
    # Core architecture
    NUM_LAYERS = 6
    HIDDEN_DIM = 48
    DROPOUT_RATE = 0.253
    ATTENTION_FREQ = 3  # Apply global attention every N layers
    
    # Model name
    NAME = "FCLGA_GraphTransformer"


class TrainingConfig:
    """Training hyperparameters"""
    
    # Training duration
    EPOCHS = 500
    
    # Batch settings
    BATCH_SIZE = 4
    
    # Optimizer
    LEARNING_RATE = 8.24e-4
    WEIGHT_DECAY = 1.07e-5
    OPTIMIZER = "adam"  # Options: adam, sgd, adamw
    
    # Learning rate scheduler
    SCHEDULER = "step"  # Options: step, cosine, plateau, none
    DECAY_STEP = 46
    DECAY_RATE = 0.668
    RESTART = 0
    
    # Dataset split
    TRAIN_SIZE = 400
    VAL_SIZE = 0  # Set to 0 to use test set for validation
    TEST_SIZE = 100
    SHUFFLE = True
    
    # Checkpointing
    SAVE_BEST_MODEL = True
    CHECKPOINT_FREQ = 50  # Save checkpoint every N epochs


class PreprocessingConfig:
    """Preprocessing and data generation parameters"""
    
    # Geometry generation
    N_SAMPLES = 500
    MIN_LENGTH = 100  # mm
    MAX_LENGTH = 200  # mm
    MIN_HOLE_RADIUS = 10  # mm
    MAX_HOLE_RADIUS = 20  # mm
    RADIUS_TOLERANCE = 30  # mm - minimum distance from hole to edges
    
    # Grid parameters for configuration generation
    N_POSITIONS = 25  # Number of different hole positions
    M_DISPLACEMENTS = 20  # Number of different displacement values
    
    # FEA simulation
    NUM_PARALLEL_JOBS = 4  # Number of parallel Abaqus simulations
    
    # Result extraction
    OUTPUT_VARIABLES = ['E11']  # Options: E11, E22, E33, S11, S22, etc.
    # For multiple variables: OUTPUT_VARIABLES = ['E11', 'E22', 'S11']


class PathConfig:
    """Directory paths"""
    
    # Data directories
    DATASET_DIR = "./datasets"
    INP_DIR = "./INPs"
    ODB_DIR = "./ODBs"
    ODB_ONLY_DIR = "./ODBsONLY"
    STRAINS_DIR = "./strains"
    
    # Output directories
    RESULTS_DIR = "./results"
    CHECKPOINT_DIR = "./results/checkpoints"
    PLOTS_DIR = "./results/plots"
    LOGS_DIR = "./results/logs"


class DeviceConfig:
    """Device configuration"""
    
    # Device selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # For manual override:
    # DEVICE = "cpu"  # Force CPU
    # DEVICE = "cuda:0"  # Specific GPU


class OptimizationConfig:
    """Hyperparameter optimization settings (for Optuna)"""
    
    # Optuna study settings
    N_TRIALS = 100
    TIMEOUT = 3600  # seconds
    
    # Search space
    NUM_LAYERS_RANGE = (4, 8)
    HIDDEN_DIM_RANGE = (32, 128)
    DROPOUT_RANGE = (0.1, 0.5)
    LR_RANGE = (1e-5, 1e-3)
    ATTENTION_FREQ_RANGE = (2, 6)


# Helper function to get all configs as a dictionary
def get_all_configs():
    """Return all configurations as a dictionary"""
    return {
        'model': {k: v for k, v in vars(ModelConfig).items() if not k.startswith('_')},
        'training': {k: v for k, v in vars(TrainingConfig).items() if not k.startswith('_')},
        'preprocessing': {k: v for k, v in vars(PreprocessingConfig).items() if not k.startswith('_')},
        'paths': {k: v for k, v in vars(PathConfig).items() if not k.startswith('_')},
        'device': {k: v for k, v in vars(DeviceConfig).items() if not k.startswith('_')},
        'optimization': {k: v for k, v in vars(OptimizationConfig).items() if not k.startswith('_')},
    }


# Helper function to print configuration
def print_config():
    """Print all configuration settings"""
    print("\n" + "="*60)
    print("FCLGA GraphTransformer Configuration")
    print("="*60)
    
    configs = get_all_configs()
    for section, params in configs.items():
        print(f"\n{section.upper()}:")
        print("-" * 40)
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    # Test: Print all configurations
    print_config()
