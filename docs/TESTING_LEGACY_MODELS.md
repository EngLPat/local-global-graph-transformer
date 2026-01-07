# Testing Legacy Model Weights

This guide explains how to test model weights trained with the legacy training script using the refactored codebase.

## Quick Start

```bash
# Basic usage with defaults (num_layers=5, hidden_dim=128, attention_freq=4)
./test_legacy_model.sh results/best_models/your_model.pt

# Or use Python directly
python src/evaluation/test_legacy_weights.py \
    --model_path results/best_models/your_model.pt \
    --num_layers 5 \
    --hidden_dim 128 \
    --attention_freq 4
```

## What It Does

The testing script performs **exactly** the same operations as `Testing_MeshGraphNets_2_optmized_svg_separated_NEWRMSE_newattention.py`:

1. **Loads your legacy model weights** - Compatible with models trained using the old script
2. **Benchmarks inference time** - Measures prediction speed and compares with FEM
3. **Evaluates on test set** - Calculates detailed RMSE metrics with masking
4. **Generates visualizations** - Creates strain field plots for test samples

## Command-Line Arguments

### Required
- `--model_path`: Path to your `.pt` checkpoint file

### Model Hyperparameters (must match training)
- `--num_layers`: Number of message passing layers (default: 5)
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--attention_freq`: Global attention frequency (default: 4)
- `--dropout_rate`: Dropout rate (default: 0.21887774707222715)

### Dataset Parameters
- `--train_size`: Training set size for splitting (default: 400)
- `--test_size`: Test set size (default: 100)
- `--batch_size`: Batch size (default: 4)

### Testing Options
- `--num_benchmark_runs`: Inference timing runs (default: 100)
- `--visualize_samples`: Number of samples to visualize (default: 25)

## Example: Testing Your Model

If you trained a model with these parameters from the legacy script:
```python
best_params = {
    'num_layers': 5,
    'attention_freq': 4,
    'hidden_dim': 128,
    'dropout_rate': 0.21887774707222715,
    'train_size': 400,
    'test_size': 100,
}
```

And saved it as: `results/0_standard_20251119_181333/best_models/model_nl5_bs4_hd128_ep3000_wd9.727199495994628e-05_lr0.0003884856764273311_shuff_True.pt`

Test it with:
```bash
python src/evaluation/test_legacy_weights.py \
    --model_path results/0_standard_20251119_181333/best_models/model_nl5_bs4_hd128_ep3000_wd9.727199495994628e-05_lr0.0003884856764273311_shuff_True.pt \
    --num_layers 5 \
    --hidden_dim 128 \
    --attention_freq 4 \
    --dropout_rate 0.21887774707222715 \
    --train_size 400 \
    --test_size 100
```

## Outputs

The script generates the same outputs as the legacy testing script:

### 1. Console Output
- Per-sample RMSE values
- Global RMSE across all samples
- Mean RMSE statistics
- Inference time benchmarks
- FEM speedup analysis

### 2. Files in `results/plots/`
- `test_results_detailed.txt` - Complete RMSE metrics
- `inference_benchmark.txt` - Timing and speedup analysis
- `test_representative_sample_X_results/` - Visualization plots
- `test_sample_N_results/` - Individual sample visualizations

### 3. Visualizations
Each visualization includes:
- Actual strain field
- Predicted strain field
- Nominal error field
- Relative error percentage

## Architecture Compatibility

✅ **The refactored model (`FCLGA_GraphTransformer`) has identical architecture to the legacy model (`MeshGraphNet`)**

This means:
- State dict keys match exactly
- Forward pass logic is identical
- Same normalization/unnormalization
- Same loss calculation

Your legacy weights will load without any modifications needed.

## Preserved Functionality

The script preserves **100% of legacy functionality**:

- ✅ Same RMSE calculation with masking for padded nodes
- ✅ Same inference benchmarking methodology
- ✅ Same visualization format (4-panel strain fields)
- ✅ Same file naming conventions
- ✅ Same statistical analysis
- ✅ Same break-even analysis for training ROI

## Troubleshooting

### "Model checkpoint not found"
- Verify the path to your `.pt` file is correct
- Use absolute path or relative to project root

### "Incompatible state dict"
- Ensure hyperparameters match your training:
  - `--num_layers` must match
  - `--hidden_dim` must match
  - `--attention_freq` must match
- Check if you used the optuna version (not frequency version)

### "Dataset not found"
- Ensure `datasets/processed_data.pt` exists
- Run preprocessing if needed: `python src/preprocessing/fclga_prepare_training_data.py`

## Notes

1. **Random Seeds**: The script uses the same seeds as legacy (`torch.manual_seed(5)`) for reproducibility

2. **Dataset Split**: Must provide same `train_size` as used during training to ensure correct test set selection

3. **Device**: Automatically uses CUDA if available, otherwise CPU (same as legacy)

4. **Normalization**: Uses the same statistics calculated from the full dataset

## Differences from Legacy Script

**None.** This script is functionally identical to the legacy testing script, just cleaner and using refactored modules.
