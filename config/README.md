# Configuration Management

## Philosophy

**Hybrid approach**: YAML file provides sensible defaults, CLI arguments allow per-experiment overrides.

```
defaults.yaml → argparse defaults → CLI overrides → Final config
```

## Usage

### 1. Edit Defaults (Development)

Open `config/defaults.yaml` and modify hyperparameters:

```yaml
model:
  num_layers: 8      # Change from 6 to 8
  hidden_dim: 64     # Change from 48 to 64

training:
  epochs: 1000       # Change from 500 to 1000
  learning_rate: 1e-3  # Change from 8.24e-4

preprocessing:
  geometry:
    n_samples: 1000        # Generate more samples
    n_hole_configs: 50     # More hole positions
    expected_elements: 2000  # Finer mesh
  
  material:
    laminate_layup: [0, 90, 90, 0]  # Change from [45, -45, -45, 45]
    ply_thickness: 0.3              # Thicker plies
```

Then run with defaults:
```bash
conda run -n fclga python -m src.training.fclga_train_model --material_type nonlinear

```

### 2. Override via CLI (Experiments)

Keep defaults unchanged, override for specific runs:

```bash
# Quick test with 2 epochs
conda run -n fclga python -m src.training.fclga_train_model --material_type nonlinear --epochs 2

# Ablation study: test different architectures
conda run -n fclga python -m src.training.fclga_train_model --material_type nonlinear --num_layers 4 --hidden_dim 32
conda run -n fclga python -m src.training.fclga_train_model --material_type nonlinear --num_layers 8 --hidden_dim 128
```

### 3. Custom Config File (Optional)

For major experiments, create custom YAML:

```bash
# Copy and edit
cp config/defaults.yaml config/experiment_deep.yaml
# Edit experiment_deep.yaml...

# Use custom config
conda run -n fclga python -m src.training.fclga_train_model --material_type nonlinear --config config/experiment_deep.yaml
```

## Configuration Files

### `defaults.yaml`
**Source of truth** for all hyperparameters. Edit this file to change default training/testing behavior.

**Note on preprocessing**: Geometry generation (`fclga_generate_geometry.py`) runs inside Abaqus's Python interpreter, which cannot load YAML. For this script, edit parameters directly in the file (they're documented at the top). All other preprocessing scripts load from YAML normally.

### `paths.py`
Directory structure and file paths. Used by all modules to locate data/results.

### `constants.py`
Physical constants, visualization settings, numerical tolerances. True constants that shouldn't vary per experiment.

## Best Practices

**For development**: Edit `defaults.yaml` directly  
**For experiments**: Use CLI overrides with `--help` to see all options  
**For papers**: Document exact CLI command used (reproducibility)  
**For hyperparameter search**: Write scripts that iterate over CLI arguments

## Example Workflow

```bash
# 1. Set reasonable defaults in YAML
vim config/defaults.yaml

# 2. Quick sanity check
conda run -n fclga python -m src.training.fclga_train_model  --material_type nonlinear --epochs 2

# 3. Full training (uses YAML defaults)
conda run -n fclga python -m src.training.fclga_train_model  --material_type nonlinear

# 4. Test trained model (extract params from filename)
conda run -n fclga python -m src.evaluation.fclga_test --material_type nonlinear\
    --model_path results/0_standard_*/best_models/model_nl6_*.pt \
    --num_layers 6 --hidden_dim 48
```
