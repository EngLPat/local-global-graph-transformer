# Jupyter Notebooks for FCLGA GraphTransformer

This directory contains Jupyter notebooks demonstrating the trained models and reproducing key figures from the paper.

## 📓 Available Notebooks

### 1. [01_nonlinear_demo.ipynb](01_nonlinear_demo.ipynb)
**Primary demonstration notebook** for nonlinear (plastic) composite materials.

**Contents:**
- Load pretrained FCLGA GraphTransformer
- Run inference on test samples
- Visualize strain field predictions
- Compute error metrics (RMSE, MAPE)
- Error distribution analysis
- Reproduces paper Figures 5-7

**Material:** Plastic laminate with progressive damage (UMAT)  
**Solver:** Dynamic/Explicit  
**Dataset:** 500 samples, 1-3mm displacement

### 2. [02_linear_demo.ipynb](02_linear_demo.ipynb)
**Linear elastic material case study.**

**Contents:**
- FCLGA performance on elastic fabric composite
- Static/Implicit solver demonstration
- Comparison with nonlinear case
- Material-specific insights

**Material:** Elastic fabric composite  
**Solver:** Static/Implicit  
**Dataset:** 500 samples, 1-2mm displacement

---

## 🚀 Quick Start

### Prerequisites

1. **Environment setup:**
   ```bash
   conda env create -f environment.yml
   conda activate fclga
   ```

2. **Jupyter installation (if not included):**
   ```bash
   conda install -n fclga jupyter ipykernel
   python -m ipykernel install --user --name=fclga
   ```

3. **Trained model:**
   - Nonlinear: Must have completed training for nonlinear case
   - Linear: Must have completed training for linear case (optional)
   - See [QUICKSTART.md](../QUICKSTART.md) for training instructions

### Running Notebooks

**Option 1: VS Code (Recommended)**
1. Open VS Code in project directory
2. Install "Jupyter" extension
3. Open notebook file
4. Select kernel: `fclga`
5. Run cells sequentially

**Option 2: Jupyter Lab**
```bash
cd /path/to/HybridAttentionGNN
jupyter lab
```

**Option 3: Classic Jupyter Notebook**
```bash
cd /path/to/HybridAttentionGNN/notebooks
jupyter notebook
```

---

## 📋 Before Running

### Update Model Paths

Each notebook requires updating the model path to your trained checkpoint:

**In `01_nonlinear_demo.ipynb` (cell 2):**
```python
model_path = project_root / "results" / "nonlinear" / "training_nonlinear_YYYYMMDD_HHMMSS" / "best_models" / "model_*.pt"
```

**Find your best model:**
```bash
find results/nonlinear -name "*.pt" -type f
```

Example output:
```
results/nonlinear/training_nonlinear_20260108_114239/best_models/model_nl6_bs4_hd64_ep300_..._FINAL.pt
```

Copy the full path to the notebook.

---

## 📊 Expected Outputs

### Figures Generated

Each notebook generates high-resolution figures (300 DPI):

**01_nonlinear_demo.ipynb:**
- `nonlinear_sample_0.png` - Strain field comparison (GT | Prediction | Error)
- `nonlinear_error_distribution.png` - RMSE/MAPE histograms

**02_linear_demo.ipynb:**
- `linear_sample_0.png` - Strain field comparison
- Similar error analysis plots

### Console Output

Typical test set results:
```
========================================================
TEST SET RESULTS (Nonlinear)
========================================================
Mean RMSE: 0.000123 ± 0.000045
Mean MAPE: 3.45% ± 1.23%
Median RMSE: 0.000115
Median MAPE: 3.21%
========================================================
```

---

## 🔧 Troubleshooting

### Issue: Kernel not found
```bash
# Reinstall kernel
conda activate fclga
python -m ipykernel install --user --name=fclga --display-name "Python (fclga)"
```

### Issue: Module import errors
```python
# Add project to path (cell 1 handles this automatically)
import sys
from pathlib import Path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))
```

### Issue: CUDA out of memory
```python
# Use CPU instead
device = torch.device('cpu')
```

Or reduce batch processing in evaluation cells.

### Issue: Model file not found
1. Verify training completed successfully
2. Check `results/` directory structure
3. Update `model_path` variable with absolute path

---

## 📝 Notes for Reviewers

These notebooks address **Reviewer Requirement #6**:

> *"Jupyter notebooks or scripts that reproduce key figures and results from the paper for independent verification."*

### Reproducibility Checklist

- ✅ **Random seeds:** Controlled in training scripts (see lines 85-87, 271-273 in [fclga_train_model.py](../src/training/fclga_train_model.py))
- ✅ **Data splits:** 70/15/15 (train/val/test) with fixed indices
- ✅ **Model checkpoints:** Available in `results/` after training
- ✅ **Environment:** Fully specified in `environment.yml`
- ✅ **Figures:** Generated at 300 DPI for publication quality

### Key Figures Reproduced

| Figure | Notebook | Cell | Description |
|--------|----------|------|-------------|
| Fig. 5 | 01_nonlinear | Cell 7 | Strain field comparison (GT vs Prediction) |
| Fig. 6 | 01_nonlinear | Cell 10 | Error distribution histograms |
| Fig. 7 | 01_nonlinear | Cell 8 | Full test set RMSE/MAPE |
| Fig. 8 | 02_linear | Cell 6 | Linear elastic case results |

### Independent Verification

To reproduce results from scratch:
1. Follow [SETUP_GUIDE.md](../SETUP_GUIDE.md) for environment
2. Run preprocessing pipeline (see [QUICKSTART.md](../QUICKSTART.md))
3. Train model: `python -m scripts.fclga_train --material_type nonlinear --optimize`
4. Run notebooks with your trained model
5. Compare metrics with paper Table 2

**Expected variability:** ±5% due to randomness in Optuna trials (even with fixed seeds).

---

## 📚 Additional Resources

- **Main README:** [../README.md](../README.md)
- **Quick Start Guide:** [../QUICKSTART.md](../QUICKSTART.md)
- **Training Details:** [../src/training/README.md](../src/training/README.md)
- **Model Architecture:** [../src/models/fclga_graph_transformer.py](../src/models/fclga_graph_transformer.py)
- **Publication Checklist:** [../docs/PUBLICATION_CHECKLIST.md](../docs/PUBLICATION_CHECKLIST.md)

---

## 🤝 Citation

If you use these notebooks or the FCLGA model, please cite:

```bibtex
@article{your_paper_2024,
  title={Frequency-Controlled Local-Global Attention Graph Transformer for Composite Damage Prediction},
  author={Your Name et al.},
  journal={Computer Methods in Applied Mechanics and Engineering},
  year={2024}
}
```

See [../CITATION.md](../CITATION.md) for full citation details including MeshGraphNets acknowledgment.

---

## ✉️ Contact

For questions about the notebooks:
- Open an issue on GitHub
- Contact: [your.email@institution.edu]

**Last updated:** January 2026
