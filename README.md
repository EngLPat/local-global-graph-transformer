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
FCLGA_GraphTransformer/
 src/
    preprocessing/       # Data generation and preprocessing
    models/              # Model architectures
    training/            # Training logic
    evaluation/          # Testing and metrics
    utils/               # Utility functions
 scripts/                 # Executable scripts
 config/                  # Configuration files
 tests/                   # Unit tests
 legacy/                  # Original working code (backup)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Abaqus (for FEA simulations)
- CUDA-capable GPU (recommended)

For detailed Abaqus configuration and material properties, see [`docs/ABAQUS_SETUP.md`](docs/ABAQUS_SETUP.md).

## Installation

### Option 1: Using Conda (Recommended)

```bash
git clone https://github.com/yourusername/FCLGA_GraphTransformer.git
cd FCLGA_GraphTransformer

# Create and activate conda environment
conda env create -f environment.yml
conda activate fclga
```

### Option 2: Using pip + virtualenv

```bash
git clone https://github.com/yourusername/FCLGA_GraphTransformer.git
cd FCLGA_GraphTransformer

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Quick Start

```bash
# Full pipeline
python scripts/fclga_run_pipeline.py --stage all --num_cpus 4

# Train only
python scripts/fclga_train.py --epochs 500 --batch_size 4

# Test only
python scripts/fclga_test.py --model_path results/best_model.pt
```

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
