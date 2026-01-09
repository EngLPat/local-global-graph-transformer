# Jupyter Notebooks for FCLGA GraphTransformer

This directory contains Jupyter notebooks demonstrating the trained models and reproducing key figures from the paper.

## 📓 Available Notebooks

### 1. [01_nonlinear_demo.ipynb](01_nonlinear_demo.ipynb)

**Contents:**
- Load pretrained FCLGA GraphTransformer
- Run inference on test samples
- Visualize strain field predictions
- Compute error metrics (RMSE, MAPE)
- Error distribution analysis
- Reproduces paper Results Figures

**Material:** nonlinear with progressive damage (UMAT)  
**Solver:** Dynamic/Explicit  
**Dataset:** 500 samples, 1-3mm displacement

### 2. [02_linear_demo.ipynb](02_linear_demo.ipynb)
**Linear elastic material case study.**

**Contents:**
- FCLGA performance on elastic fabric composite
- Static/Implicit solver demonstration
- Comparison with nonlinear case
- Material-specific insights

**Material:** Linear Elastic  
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

### Running Notebooks

**Option 1: VS Code (Recommended)**
1. Open VS Code in project directory
2. Install "Jupyter" extension
3. Open notebook file
4. Select kernel: `fclga`
5. Run cells sequentially

**Option 2: Jupyter Lab**
```bash
cd /path/to/local-global-graph-transformer
jupyter lab
```

**Option 3: Classic Jupyter Notebook**
```bash
cd /path/to/local-global-graph-transformer/notebooks
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
# Use CPU instead (not recommended)
device = torch.device('cpu')
```

Or reduce batch processing in evaluation cells. (recommended)

### Issue: Model file not found
1. Verify training completed successfully
2. Check `results/` directory structure
3. Update `model_path` variable with absolute path
