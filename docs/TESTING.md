# Testing Guide

This guide explains how to test trained FCLGA GraphTransformer models.

## Testing Script

Use `src/evaluation/fclga_test.py` to evaluate trained models:

```bash
python -m src.evaluation.fclga_test \
    --model_path path/to/model.pt \
    --dataset_path path/to/dataset.pt \
    --num_layers 5 \
    --hidden_dim 128 \
    --attention_freq 4 \
    --train_size 400 \
    --visualize_samples 25
```

## Required Arguments

- `--model_path`: Path to trained model checkpoint (.pt file)

## Optional Arguments

### Model Configuration (must match training)
- `--num_layers`: Number of message passing layers (default: 5)
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--attention_freq`: Global attention frequency (default: 4)
- `--dropout_rate`: Dropout rate (default: 0.219)

### Dataset Configuration
- `--dataset_path`: Path to dataset file (default: datasets/processed_data.pt)
- `--train_size`: Training set size for splitting (default: 400)
- `--batch_size`: Batch size (default: 4)

### Testing Options
- `--num_benchmark_runs`: Number of runs for inference benchmarking (default: 100)
- `--visualize_samples`: Number of test samples to visualize (default: 25)

## Output

The testing script generates:

1. **Performance Metrics**
   - Global RMSE (all nodes, all samples)
   - Per-sample RMSE statistics
   - R² scores for representative samples
   - Loss metrics

2. **Inference Benchmarks**
   - Average inference time
   - Speedup vs FEM (Abaqus)
   - Break-even analysis
   - Results saved to `results/plots/inference_benchmark.txt`

3. **Visualizations** (saved to `results/plots/`)
   - Actual strain fields
   - Predicted strain fields
   - Nominal error plots
   - Relative error plots
   - Regression plots (predicted vs actual)

4. **Detailed Results**
   - Per-sample RMSE values
   - Masking statistics (padded nodes)
   - Comprehensive metrics file: `results/plots/test_results_detailed.txt`

## Example: Testing Legacy Model

```bash
python -m src.evaluation.fclga_test \
    --model_path legacy/model_nl5_bs4_hd128_ep3000_*.pt \
    --dataset_path legacy/processed_data.pt \
    --train_size 400 \
    --visualize_samples 5
```

## Dataset Splitting

The script automatically:
1. Shuffles the dataset with seed=5 (for reproducibility)
2. Splits into train/val/test sets
3. Uses training set for normalization statistics

This matches the behavior of the legacy training code.

## Understanding Results

### Global RMSE
Average error across all nodes and all test samples. For well-trained models on composite plate problems, expect RMSE ≈ 0.002.

### Per-Sample RMSE
RMSE calculated for each test sample individually. Check the range to identify outliers.

### Masking
The model handles variable-sized graphs by padding. The masking percentage indicates how many nodes are padded (typically 3-5%).

### Inference Speedup
Compared to Abaqus FEM (≈120 seconds per simulation). Typical speedup: 10,000-25,000×.

### Break-even Point
Number of predictions needed to recover training time investment. For GNN models, typically <50 simulations.

## Troubleshooting

### FileNotFoundError
- Check that model and dataset paths are correct
- Use absolute paths or paths relative to project root

### Shape Mismatch Errors
- Verify model hyperparameters match training configuration
- Check `--num_layers`, `--hidden_dim`, `--attention_freq`

### Poor Performance (High RMSE)
- Ensure dataset is shuffled (automatic in script)
- Verify model was trained on same data distribution
- Check that normalization statistics are computed from training set

### CUDA Out of Memory
- Reduce `--batch_size`
- Reduce `--visualize_samples`
- Test on CPU by modifying device selection

## Code Structure

### Main Components

1. **parse_arguments()**: Parse command-line arguments
2. **load_and_split_dataset()**: Load and split dataset with shuffling
3. **create_model()**: Initialize model architecture
4. **load_model_weights()**: Load trained weights
5. **main()**: Orchestrate testing workflow

### Utility Functions (src/utils/test_utils.py)

- **visualize_sample()**: Generate strain field plots
- **evaluate_model()**: Calculate RMSE and metrics
- **benchmark_inference()**: Time model predictions
- **ObjectView**: Helper class for argument handling

## Linting and Code Quality

All testing code is linted with flake8:

```bash
flake8 src/evaluation/fclga_test.py --max-line-length=100 --ignore=W503
flake8 src/utils/test_utils.py --max-line-length=100 --ignore=W503
```

Follows Python professional standards:
- PEP 8 compliant
- Complete docstrings (Google style)
- Type hints in comments
- Proper error handling
