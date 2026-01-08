# Material Type Organization

## Overview
The codebase now supports two material cases with organized directory structure:
- **linear**: Elastic fabric composite (Static implicit solver)
- **nonlinear**: Plastic laminate (Dynamic Explicit solver)

## Directory Structure

```
src/preprocessing/
├── linear/                    # Elastic case preprocessing
│   ├── fclga_generate_geometry.py
│   ├── fclga_run_simulations.py
│   ├── fclga_extract_features.py
│   ├── fclga_extract_results.py
│   ├── fclga_build_dataset.py (TODO)
│   └── fclga_prepare_training_data.py (TODO)
│
└── nonlinear/                 # Plastic case preprocessing
    ├── fclga_generate_geometry.py
    ├── fclga_run_simulations.py
    ├── fclga_extract_features.py
    ├── fclga_extract_results.py
    ├── fclga_build_dataset.py
    └── fclga_prepare_training_data.py

data/
├── raw/
│   ├── linear/
│   │   ├── geometry/          # .inp files
│   │   └── simulations/       # .odb files  
│   └── nonlinear/
│       ├── geometry/
│       └── simulations/
├── interim/
│   ├── linear/
│   │   └── strains/           # E11_*.txt files
│   └── nonlinear/
│       └── strains/
└── processed/
    ├── linear/
    │   ├── datasets/
    │   │   └── processed_data.pt
    │   ├── node_gnn_data.pt
    │   ├── strains.pt
    │   └── triangulation_data.pkl
    └── nonlinear/
        ├── datasets/
        │   └── processed_data.pt
        ├── node_gnn_data.pt
        ├── strains.pt
        └── triangulation_data.pkl

results/
├── linear/
│   ├── images/                # Geometry visualizations
│   └── training_linear_YYYYMMDD_HHMMSS/
│       ├── best_models/
│       └── training_results/
└── nonlinear/
    ├── images/
    └── training_nonlinear_YYYYMMDD_HHMMSS/
        ├── best_models/
        └── training_results/
```

## Command Examples

### Nonlinear (Plastic) Case

```bash
# Preprocessing pipeline
python -m src.preprocessing.nonlinear.fclga_generate_geometry
python -m src.preprocessing.nonlinear.fclga_run_simulations
abaqus python src/preprocessing/nonlinear/fclga_extract_results.py
python -m src.preprocessing.nonlinear.fclga_extract_features
python -m src.preprocessing.nonlinear.fclga_build_dataset
python -m src.preprocessing.nonlinear.fclga_prepare_training_data

# Training with hyperparameter optimization
python -m src.training.fclga_train_model \
    --material_type nonlinear \
    --optimize \
    --optuna_trials 50 \
    --epochs 50 \
    --final_epochs 500

# Direct training (no optimization)
python -m src.training.fclga_train_model \
    --material_type nonlinear \
    --num_layers 7 \
    --batch_size 4 \
    --hidden_dim 96 \
    --epochs 600

# Testing
python -m src.evaluation.fclga_test \
    --material_type nonlinear \
    --training_run results/nonlinear/training_nonlinear_20260107_152157
```

### Linear (Elastic) Case

```bash
# Preprocessing pipeline
python -m src.preprocessing.linear.fclga_generate_geometry
python -m src.preprocessing.linear.fclga_run_simulations
abaqus python src/preprocessing/linear/fclga_extract_results.py
python -m src.preprocessing.linear.fclga_extract_features
python -m src.preprocessing.linear.fclga_build_dataset (TODO)
python -m src.preprocessing.linear.fclga_prepare_training_data (TODO)

# Training
python -m src.training.fclga_train_model \
    --material_type linear \
    --optimize \
    --optuna_trials 50 \
    --epochs 50 \
    --final_epochs 500

# Testing  
python -m src.evaluation.fclga_test \
    --material_type linear \
    --training_run results/linear/training_linear_YYYYMMDD_HHMMSS
```

## Key Changes

### 1. Preprocessing Organization
- Each material type has its own preprocessing folder
- Same filenames in different folders (namespace separation)
- All scripts updated to use `data/{raw,interim,processed}/{material_type}/` paths

### 2. Training Script (`src/training/fclga_train_model.py`)
- Added `--material_type` argument (default: "nonlinear")
- Automatically loads data from `data/processed/{material_type}/datasets/processed_data.pt`
- Creates training runs in `results/{material_type}/training_{material_type}_TIMESTAMP/`

### 3. Testing Script (`src/evaluation/fclga_test.py`)
- Added `--material_type` argument (default: "nonlinear")
- Automatically loads dataset from correct material type folder
- Works with new training run naming convention

### 4. Config Module (`config/paths.py`)
- New `get_paths(material_type)` function returns material-specific paths
- Maintains backward compatibility with legacy code (defaults to nonlinear)

### 5. Training Utilities (`src/utils/training_utils.py`)
- `create_results_folder(material_type)` creates organized result directories
- Training run naming: `training_{material_type}_TIMESTAMP`

## Benefits

1. **Clear separation**: Linear and nonlinear cases are completely separated
2. **No confusion**: Training results clearly labeled by material type  
3. **Scalability**: Easy to add more material types in the future
4. **Professional**: Follows ML best practices (e.g., Hugging Face, PyTorch)
5. **Parallel work**: Can work on both cases without conflicts
6. **Data organization**: All data properly organized by processing stage and type

## Migration Notes

Existing data has been moved:
- `data/processed/datasets/processed_data.pt` → `data/processed/nonlinear/datasets/processed_data.pt`
- `results/training_run_20260107_152157/` → `results/nonlinear/training_nonlinear_20260107_152157/`

## TODO for Linear Case

Complete the preprocessing pipeline:
1. Create `fclga_build_dataset.py` (adapt from nonlinear)
2. Create `fclga_prepare_training_data.py` (adapt from nonlinear)
3. Run full linear preprocessing pipeline
4. Train model on linear dataset
5. Compare linear vs nonlinear results
