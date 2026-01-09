# Scripts Directory

Main entry points and utilities for the FCLGA GraphTransformer project.

## Main Pipeline

### `fclga_run_pipeline.py`
Complete end-to-end pipeline orchestrator.

```bash
# Run complete preprocessing (nonlinear)
python scripts/fclga_run_pipeline.py --stage preprocess --material_type nonlinear

# Run with specific CPU count
python scripts/fclga_run_pipeline.py --stage preprocess --material_type nonlinear --num_cpus 8

# Run everything (preprocess + train + test)
python scripts/fclga_run_pipeline.py --stage all --material_type nonlinear

# Linear elastic material
python scripts/fclga_run_pipeline.py --stage all --material_type linear
```

**Features:**
- ✅ Automatic directory creation (organized data structure)
- ✅ Progress tracking and error handling
- ✅ Temporary file cleanup
- ✅ Validation after each step

## Validation and Testing

### `validate_processed_data.py`
Validates preprocessed data and generates visualizations.

```bash
# Validate data and create plots
python scripts/validate_processed_data.py

# Visualize specific samples
python scripts/validate_processed_data.py --sample 5 --num-samples 5

# Just check data (no plots)
python scripts/validate_processed_data.py --no-plots
```

**Generates:**
- Mesh geometry plots
- Strain field visualizations  
- Dataset statistics
- Data consistency reports

**Output:** `results/plots/validation/`

## Utilities

### `cleanup_temp_files.sh`
Cleans up Abaqus temporary files from project root.

```bash
bash scripts/cleanup_temp_files.sh
```

**Removes:**
- `*.stt`, `*.res`, `*.pac`, `*.mdl`, `*.abq`, `*.lck`
- `abaqus.rpy*`, `*.rec`

**Note:** The pipeline now automatically handles cleanup, but this is useful for manual cleanup of existing files.

## Training and Testing

### `fclga_train.py`
(To be implemented - training script)

### `fclga_test.py`
(To be implemented - testing/evaluation script)

## Workflow

### Standard Research Workflow

1. **Run preprocessing:**
   ```bash
   python scripts/fclga_run_pipeline.py --stage preprocess
   ```

2. **Validate results:**
   ```bash
   python scripts/validate_processed_data.py
   ```

3. **Check plots:**
   Open files in `results/plots/validation/`

4. **Train model:**
   ```bash
   python scripts/fclga_train.py
   ```

5. **Evaluate model:**
   ```bash
   python scripts/fclga_test.py
   ```

### Troubleshooting

**"ModuleNotFoundError":**
```bash
conda activate fclga  # Make sure you're in the right environment
```

**Temporary files cluttering root:**
```bash
bash scripts/cleanup_temp_files.sh
```

**Want to reprocess specific steps:**
- Edit individual preprocessing scripts in `src/preprocessing/`
- Or delete specific output directories and rerun pipeline

## Best Practices

- **Always activate environment:** `conda activate fclga` before running scripts
- **Check validation plots:** After preprocessing, always run validation
- **Monitor disk space:** Simulations generate large .odb files
- **Use version control:** Commit code changes before long runs
