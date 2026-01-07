# Quick Start Guide - FCLGA GraphTransformer

Welcome! This guide will get you up and running in minutes.

## 📋 Prerequisites

- **Linux/Mac/Windows** with terminal access
- **Conda** or **Miniconda** installed ([download here](https://docs.conda.io/en/latest/miniconda.html))
- **Abaqus** (optional - only for data generation)
- **Git** for cloning the repository

## 🚀 Getting Started (5 Minutes)

### Step 1: Create Environment

```bash
# This reads environment.yml and installs all dependencies
conda env create -f environment.yml
```

**What this does:** Creates a new conda environment called `fclga` with Python 3.10, PyTorch, PyTorch Geometric, NumPy, Matplotlib, and all other required packages.

⏱️ **Time:** 5-10 minutes depending on your internet connection

### Step 2: Activate Environment

```bash
conda activate fclga
```

### Step 4: Verify Installation

```bash
python test_environment.py
```

You should see:
```
✓ Python version: 3.10.x
✓ PyTorch available
✓ PyTorch Geometric available
✓ NumPy available
✓ All dependencies satisfied!
```

## 🎯 Running Your First Pipeline

### Option 1: With Abaqus (Full Pipeline)

If you have Abaqus installed:

```bash
# Make sure environment is activated
conda activate fclga

# Run preprocessing (generates data from scratch)
python scripts/fclga_run_pipeline.py --stage preprocess --num_cpus 4

# This will take a while (hours for 500 samples)
# For a quick test with 10 samples, edit src/preprocessing/fclga_generate_geometry.py:
# Change N = 2, M = 5 to N = 2, M = 5 (already set for quick test!)
```

### Option 2: With Preprocessed Data (Skip Data Generation)

If you have preprocessed data files or don't have Abaqus:

```bash
# Place your preprocessed data in:
data/processed/
├── node_gnn_data.pt
├── strains.pt
├── triangulation_data.pkl
└── plate_geometry_data.pt (optional)

# Then validate it
python scripts/validate_processed_data.py
```

## 📊 Validate Your Data

After preprocessing, always validate:

```bash
conda activate fclga
python scripts/validate_processed_data.py --num-samples 5
```

**Output:** Beautiful plots in `results/plots/validation/`
- `sample_000_mesh.png` - Shows mesh geometry
- `sample_000_strain_field.png` - Shows strain distribution
- `dataset_statistics.png` - Dataset overview

## 💡 Common Workflows

### Daily Workflow

```bash
# Start your work session
cd FCLGA_GraphTransformer
conda activate fclga

# Run your commands...
python scripts/validate_processed_data.py

# When done
conda deactivate
```

### Cleaning Up Temporary Files

```bash
bash scripts/cleanup_temp_files.sh
```

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

**Problem:** Environment not activated

**Solution:**
```bash
conda activate fclga
# Check that (fclga) appears in your prompt
```

### "conda: command not found"

**Problem:** Conda not installed or not in PATH

**Solution:** 
1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Restart your terminal
3. Try again

### Environment activation doesn't stick

**Solution:** Add to your `~/.bashrc` or `~/.zshrc`:
```bash
# Auto-activate fclga when entering project directory
cd() {
    builtin cd "$@"
    if [[ -f environment.yml ]] && [[ $PWD == */FCLGA_GraphTransformer ]]; then
        conda activate fclga
    fi
}
```

### Forgot which environment I'm in?

```bash
conda env list
# The active environment has a * next to it
```

## 🔄 Updating Your Environment

If `environment.yml` changes:

```bash
conda activate fclga
conda env update -f environment.yml --prune
```

## 📚 Next Steps

1. ✅ Read the main [README.md](README.md) for detailed documentation
2. ✅ Check [scripts/README.md](scripts/README.md) for script usage
3. ✅ See [TESTING_PLAN.md](TESTING_PLAN.md) for preprocessing details
4. ✅ Review [data/README.md](data/README.md) for data organization

## 💬 Need Help?

- Check [TESTING_PLAN.md](TESTING_PLAN.md) for detailed preprocessing info
- Review error messages carefully - they often tell you exactly what's wrong
- Make sure `(fclga)` is in your prompt before running Python scripts

## ✨ Quick Reference Card

```bash
# Setup (once)
conda env create -f environment.yml

# Daily use (every time)
conda activate fclga

# Run preprocessing
python scripts/fclga_run_pipeline.py --stage preprocess

# Validate data
python scripts/validate_processed_data.py

# Clean up
bash scripts/cleanup_temp_files.sh

# Exit
conda deactivate
```

---

**Remember:** Always activate the environment before running scripts!

```bash
conda activate fclga  # See (fclga) in prompt
python scripts/...    # Now you can run Python scripts
```

1. ✅ Professional folder structure
2. ✅ All original files backed up safely
3. ✅ Git configuration ready
4. ✅ Documentation framework
5. ✅ Clear separation of concerns
6. ✅ Ready for incremental refactoring

## Risk Level: ZERO

- No code has been modified
- No logic has been changed
- Everything is reversible
- Original files are untouched

## Questions to Consider

Before Phase 2:

1. Are you comfortable with the structure?
2. Do you want to adjust any folder names?
3. Should we add anything else before copying files?
4. Ready to proceed with file copying (still no code changes)?

## Contact & Support

If anything is unclear or you want to discuss the approach, just ask!

---

**Next Command (When Ready):**
"Let's proceed with Phase 2A - copying files with new names"

**Or:**
"I want to review/adjust something first"
