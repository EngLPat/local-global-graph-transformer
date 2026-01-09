# FCLGA GraphTransformer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Graph Neural Network with Hybrid Local-Global Attention for predicting mechanical response in structures.

**Paper:** "Graph Neural Networks with Hybrid Local-Global Attention for Effective Prediction of Mechanical Response in Structures"  
**Authors:** Luca Patrignani, Silvestre T. Pinho | Imperial College London

## Repository Structure

```
HybridAttentionGNN/
├── config/
│   ├── defaults.yaml          # Edit simulation/training parameters here
│   ├── paths.py               # Automatic path management (linear/nonlinear)
│   └── constants.py           # Physical constants
├── src/
│   ├── preprocessing/
│   │   ├── linear/            # Elastic fabric composite (Static/Implicit)
│   │   └── nonlinear/         # Plastic laminate (Dynamic/Explicit)
│   ├── models/                # FCLGA GraphTransformer architecture
│   ├── training/              # Training with Optuna hyperparameter tuning
│   ├── evaluation/            # Testing and visualization
│   └── utils/                 # Shared utilities
├── data/
│   ├── raw/{linear,nonlinear}/
│   │   ├── geometry/          # .inp files from Abaqus
│   │   ├── simulations/       # .odb files from Abaqus
│   │   └── geometry_images/   # Geometry visualization
│   ├── interim/{linear,nonlinear}/
│   │   └── strains/           # Extracted strain values
│   └── processed/{linear,nonlinear}/
│       └── datasets/          # Final graph datasets for training
├── results/{linear,nonlinear}/
│   └── training_*_TIMESTAMP/
│       ├── best_models/       # Saved checkpoints
│       └── training_results/  # Loss curves, hyperparameters
└── notebooks/                 # Jupyter notebooks reproducing key figures
    ├── 01_nonlinear_demo.ipynb
    ├── 02_linear_demo.ipynb
    └── README.md
```

## Requirements

- **Python 3.10+** with PyTorch 2.0+, PyTorch Geometric
- **Abaqus**
- **CUDA GPU** (recommended for training)

## Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/HybridAttentionGNN.git
cd HybridAttentionGNN

# 2. Create conda environment
conda env create -f environment.yml
conda activate fclga

# 3. Verify installation
python -c "import torch; import torch_geometric; print('✓ Ready')"
```

## Usage

### Material Types

- **`linear`**: Elastic fabric composite, Static/Implicit solver, 1-2mm displacement
- **`nonlinear`**: Plastic laminate, Dynamic/Explicit solver, 1-3mm displacement

All commands support `--material_type {linear,nonlinear}` (default: nonlinear).

### Preprocessing Pipeline

Generate 500 samples with varying geometry and loads:

```bash
# === NONLINEAR (default) ===
# 1. Generate 500 geometries (~5 min)
abaqus cae nogui=src/preprocessing/nonlinear/fclga_generate_geometry.py

# 2. Run simulations in parallel (~1-4 hours, configure workers in config/defaults.yaml line 77)
python -m src.preprocessing.nonlinear.fclga_run_simulations

# 3. Extract strains from ODB files (~10 min)
abaqus cae nogui=src/preprocessing/nonlinear/fclga_extract_results.py

# 4. Extract graph features (~5 min)
python -m src.preprocessing.nonlinear.fclga_extract_features

# 5. Build strain dataset (~2 min)
python -m src.preprocessing.nonlinear.fclga_build_dataset

# 6. Prepare training data (~5 min)
python -m src.preprocessing.nonlinear.fclga_prepare_training_data

# === LINEAR (elastic) ===
# Same commands, replace 'nonlinear' with 'linear'
abaqus cae nogui=src/preprocessing/linear/fclga_generate_geometry.py
python -m src.preprocessing.linear.fclga_run_simulations
abaqus cae nogui=src/preprocessing/linear/fclga_extract_results.py
python -m src.preprocessing.linear.fclga_extract_features
python -m src.preprocessing.linear.fclga_build_dataset
python -m src.preprocessing.linear.fclga_prepare_training_data
```

**Configuration:** Edit [`config/defaults.yaml`](config/defaults.yaml) line 77 to change parallel simulation workers (default: 4).

### Training

Train with Optuna hyperparameter optimization:

```bash
# Nonlinear (progressive damage)
python -m src.training.fclga_train_model --material_type nonlinear

# Linear (elastic)
python -m src.training.fclga_train_model --material_type linear

# Custom hyperparameters
python -m src.training.fclga_train_model \
    --material_type nonlinear \
    --num_layers 6 \
    --hidden_dim 48 \
    --attention_freq 3 \
    --learning_rate 8.24e-4 \
    --epochs 3000

# Custom data split (default is 70/15/15)
python -m src.training.fclga_train_model \
    --material_type nonlinear \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

**Data Split:** By default uses **70/15/15** (train/val/test). Configure via `--train_ratio`, `--val_ratio`, `--test_ratio` or edit `config/defaults.yaml`.

Results saved to `results/{material_type}/training_{material_type}_TIMESTAMP/`.

### Testing

Evaluate trained models:

```bash
# Test best model from training run
python -m src.evaluation.fclga_test \
    --model_path results/nonlinear/training_nonlinear_20260107_152157/best_models/model_nl5_bs4_*.pt \
    --material_type nonlinear

# Test with specific material type
python -m src.evaluation.fclga_test \
    --model_path results/linear/training_linear_*/best_models/*.pt \
    --material_type linear
```

Generates:
- Per-sample error visualizations (PDF)
- Overall performance metrics (RMSE, MAPE)
- Saved in `results/{material_type}/training_*/test_sample_*_results.pdf`

### Interactive Notebooks

Reproduce paper figures and visualize results:

```bash
# Launch Jupyter
jupyter lab notebooks/

# Or open in VS Code with Jupyter extension
code notebooks/01_nonlinear_demo.ipynb
```

**Available notebooks:**
- **[`01_nonlinear_demo.ipynb`](notebooks/01_nonlinear_demo.ipynb)**: Nonlinear case.
- **[`02_linear_demo.ipynb`](notebooks/02_linear_demo.ipynb)**: Linear elastic case.

See [notebooks/README.md](notebooks/README.md) for detailed usage instructions.

## Key Files

- **[`ABAQUS_SETUP.md`](ABAQUS_SETUP.md)**: Complete Abaqus FEA documentation (material properties, solver settings, reproducibility)
- **[`config/defaults.yaml`](config/defaults.yaml)**: Edit geometry ranges, material properties, simulation workers, training hyperparameters
- **[`config/paths.py`](config/paths.py)**: Automatic path management for linear/nonlinear separation
- **[`src/models/fclga_graph_transformer.py`](src/models/fclga_graph_transformer.py)**: Model architecture
- **[`src/training/fclga_train_model.py`](src/training/fclga_train_model.py)**: Training loop with Optuna
- **[`src/evaluation/fclga_test.py`](src/evaluation/fclga_test.py)**: Testing and visualization

## Citation

```bibtex
@article{patrignani2025hybrid,
  title={Graph Neural Networks with Hybrid Local-Global Attention for Effective 
         Prediction of Mechanical Response in Structures},
  author={Patrignani, Luca and Pinho, Silvestre T.},
  journal={Computer Methods in Applied Mechanics and Engineering},
  year={2025}
}
```

## License

MIT License - See LICENSE file
