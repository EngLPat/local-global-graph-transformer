# FCLGA GraphTransformer

Graph Neural Network with Hybrid Local-Global Attention for predicting mechanical response in structures.

**Authors:** Luca Patrignani, Silvestre T. Pinho  
**Institution:** Imperial College London  
**Paper:** "Graph Neural Networks with Hybrid Local-Global Attention for Effective Prediction of Mechanical Response in Structures"  
**Journal:** Computer Methods in Applied Mechanics and Engineering

## Overview

This repository implements the FCLGA GraphTransformer for solving mesh-based structural mechanics problems using FEA data.

## Features

- Automated FEA dataset generation using Abaqus
- Graph-based mesh representation
- Hybrid local-global attention mechanism
- Parallel simulation execution
- Comprehensive evaluation and visualization

## Project Structure

```
HybridAttentionGNN/
├── src/
│   ├── preprocessing/       # Data generation and preprocessing
│   ├── models/              # Model architectures (FCLGA_GraphTransformer)
│   ├── training/            # Training loop and optimization
│   ├── evaluation/          # Testing and metrics
│   └── utils/               # Shared utility functions
├── config/                  # Configuration management
│   ├── defaults.yaml        # Default hyperparameters (EDIT THIS)
│   ├── paths.py             # Directory structure
│   └── constants.py         # Physical/numerical constants
├── scripts/                 # Executable entry points
├── datasets/                # Processed graph datasets
├── results/                 # Training outputs and checkpoints
├── legacy/                  # Original implementation (backup)
└── tests/                   # Unit tests
```

## Configuration

**Edit `config/defaults.yaml` to change default hyperparameters.**  
All parameters can be overridden via CLI arguments (see `--help`).

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Abaqus (for FEA simulations)
- CUDA-capable GPU (recommended)

For detailed Abaqus configuration and material properties, see [`docs/ABAQUS_SETUP.md`](docs/ABAQUS_SETUP.md).

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/FCLGA_GraphTransformer.git
cd FCLGA_GraphTransformer
```

### Step 2: Set Up Python Environment

**Option A: Using Conda (Recommended)**

```bash
# Create the environment from the provided environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate fclga

# Verify installation
python -c "import torch; import torch_geometric; print('✓ Environment ready!')"
```

**Option B: Using pip + virtualenv**

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows

# Install PyTorch and PyTorch Geometric (check pytorch.org for your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric

# Install other dependencies
pip install numpy scipy pandas matplotlib seaborn pyyaml tqdm pytest
```

### Step 3: Verify Abaqus Installation (Optional for Data Generation)

```bash
# Check if Abaqus is available
abaqus information=system

# If not found, you'll need to add Abaqus to your PATH or use existing preprocessed data
```

**Note:** Abaqus is only required for data generation (preprocessing). If you have preprocessed data, you can skip this.

### Step 4: Test the Installation

```bash
# Make sure you're in the conda environment (you should see (fclga) in your prompt)
conda activate fclga

# Run environment test
python test_environment.py

# Or quick test
python -c "import torch; import torch_geometric; import numpy; print('All packages available!')"
```

## Quick Start

### For First-Time Users (After Pulling from GitHub)

```bash
# 1. Activate the environment (IMPORTANT - do this every time!)
conda activate fclga

# 2. Generate data and run preprocessing (requires Abaqus)
python scripts/fclga_run_pipeline.py --stage preprocess --num_cpus 4

# 3. Validate the processed data
python scripts/validate_processed_data.py

# 4. Check the generated plots
ls results/plots/validation/

# 5. Train the model (coming soon)
python scripts/fclga_train.py --epochs 500 --batch_size 4

# 6. Test the model (coming soon)
python scripts/fclga_test.py --model_path results/best_model.pt
```

### Quick Commands Reference

```bash
# Always start by activating the environment
conda activate fclga

# Run full preprocessing pipeline
python scripts/fclga_run_pipeline.py --stage preprocess

# Validate preprocessed data
python scripts/validate_processed_data.py --num-samples 5

# Clean up temporary Abaqus files
bash scripts/cleanup_temp_files.sh

# Deactivate environment when done
conda deactivate
```

### Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'numpy'` (or torch, etc.)

**Solution:** You forgot to activate the conda environment!
```bash
conda activate fclga
```

**Problem:** Temporary Abaqus files (*.stt, *.res, etc.) cluttering the directory

**Solution:** Run the cleanup script
```bash
bash scripts/cleanup_temp_files.sh
```

**Problem:** `abaqus: command not found`

**Solution:** Either add Abaqus to PATH or use preprocessed data. Abaqus is only needed for data generation.

## Pipeline Stages

1. **Geometry Generation** - Create parametric FEA models
2. **FEA Simulation** - Run Abaqus simulations in parallel
3. **Feature Extraction** - Extract mesh geometry and build graph
4. **Result Extraction** - Extract nodal results (E11, E22, S11, etc.)
5. **Dataset Building** - Combine features and targets
6. **Training** - Train FCLGA GraphTransformer
7. **Testing** - Evaluate and visualize results

## Configuration

Default hyperparameters in `config/defaults.yaml` or override via CLI:

```bash
python scripts/fclga_train.py \
    --num_layers 6 \
    --hidden_dim 48 \
    --attention_freq 3 \
    --learning_rate 8.24e-4
```

## Citation

If you use this code, please cite:

```bibtex
@article{patrignani2025hybrid,
  title={Graph Neural Networks with Hybrid Local-Global Attention for Effective Prediction of Mechanical Response in Structures},
  author={Patrignani, Luca and Pinho, Silvestre T.},
  journal={Computer Methods in Applied Mechanics and Engineering},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contact

**Luca Patrignani**  
Imperial College London  
Email: l.patrignani@imperial.ac.uk

## Acknowledgments

This work was conducted at Imperial College London under the supervision of Prof. Silvestre T. Pinho.
