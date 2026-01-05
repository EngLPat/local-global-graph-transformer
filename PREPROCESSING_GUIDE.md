# Preprocessing Pipeline - Step-by-Step Guide

## Current Status: Starting Fresh

No preprocessing data exists yet. We'll create it step by step.

---

## 🚀 Quick Start Commands

### Option A: Run Full Pipeline (If you have Abaqus + time)

```bash
# Navigate to project root
cd c:\Users\lpatrign\Desktop\python_paper

# Step 1: Generate geometry (requires Abaqus) - ~10-30 minutes
abaqus cae nogui=FCLGA_GraphTransformer/src/preprocessing/fclga_generate_geometry.py

# Step 2: Run simulations (requires Abaqus) - HOURS!
python FCLGA_GraphTransformer/src/preprocessing/fclga_run_simulations.py

# Step 3: Extract features - Fast!
python FCLGA_GraphTransformer/src/preprocessing/fclga_extract_features.py

# Step 4: Copy ODB files
python FCLGA_GraphTransformer/legacy/extractodb.py

# Step 5: Extract results (requires Abaqus) - ~30 minutes
abaqus cae nogui=FCLGA_GraphTransformer/src/preprocessing/fclga_extract_results.py

# Step 6: Build dataset - Fast!
python FCLGA_GraphTransformer/src/preprocessing/fclga_build_dataset.py

# Test everything worked
cd FCLGA_GraphTransformer
python test_preprocessing.py
```

### Option B: Test with Existing Legacy Data (Recommended First!)

If you have existing data from previous runs:

```bash
cd c:\Users\lpatrign\Desktop\python_paper

# Test if legacy scripts still work
python legacy/openhole_features_composite_nodes.py
python legacy/strains_tensor_nonlinear.py

# Then compare with new scripts
python FCLGA_GraphTransformer/src/preprocessing/fclga_extract_features.py
python FCLGA_GraphTransformer/src/preprocessing/fclga_build_dataset.py
```

### Option C: Small Test Run (Recommended!)

Modify scripts to generate only 10 samples instead of 500:

```python
# In fclga_generate_geometry.py, change:
N = 2  # Instead of 25
M = 5  # Instead of 20
# Total: 2x5 = 10 samples instead of 500
```

---

## 📋 Detailed Instructions

### Step 1: Generate Geometry

**Script:** `fclga_generate_geometry.py`

**What it does:** Creates parametric plate geometries with holes

**Requirements:**
- Abaqus with CAE
- Torch in Abaqus Python environment

**Command:**
```bash
cd c:\Users\lpatrign\Desktop\python_paper
abaqus cae nogui=FCLGA_GraphTransformer/src/preprocessing/fclga_generate_geometry.py
```

**Expected Time:** 10-30 minutes (500 models)

**Output:**
- `./INPs/` directory with 500 .inp files
- `plate_geometry_data.pt`

**Check Success:**
```powershell
(Get-ChildItem ./INPs -Filter *.inp | Measure-Object).Count
# Should show 500
```

**Note:** Script currently has hard-coded paths. Works from `python_paper` directory.

---

### Step 2: Run Simulations

**Script:** `fclga_run_simulations.py`

**What it does:** Runs Abaqus FEA simulations in parallel

**Requirements:**
- Abaqus solver
- `./INPs/` from Step 1

**Command:**
```bash
python FCLGA_GraphTransformer/src/preprocessing/fclga_run_simulations.py
```

**Expected Time:** SEVERAL HOURS (depends on CPU cores)

**Configuration:**
```python
# Line ~20 in script:
num_parallel_simulations = 10  # Adjust based on your CPU cores
```

**Output:**
- `./ODBs/` directory with subdirectories for each model
- Each contains: .odb, .dat, .msg, .sta, .log files

**⚠️ WARNING:** Very time-consuming! Consider small test first.

---

### Step 3: Extract Features

**Script:** `fclga_extract_features.py`

**What it does:** Parses mesh from .inp files, creates graph structure

**Requirements:**
- Python with torch, torch_geometric
- `./INPs/` from Step 1

**Command:**
```bash
cd c:\Users\lpatrign\Desktop\python_paper
python FCLGA_GraphTransformer/src/preprocessing/fclga_extract_features.py
```

**Expected Time:** < 5 minutes

**Output:**
- `node_gnn_data.pt` - Graph data
- `triangulation_data.pkl` - For visualization

**Check Success:**
```python
import torch
data = torch.load('node_gnn_data.pt')
print(f"Loaded {len(data)} graphs")
```

**Note:** Script expects .inp files in current directory, not ./INPs/

**⚠️ Path Issue:** May need to update line ~107:
```python
# Change:
inp_files = [f for f in os.listdir(directory) if f.endswith(".inp")]
# To:
inp_files = [f for f in os.listdir("./INPs") if f.endswith(".inp")]
```

---

### Step 4: Copy ODB Files

**Script:** `extractodb.py` (from legacy)

**What it does:** Copies all .odb files to single directory

**Requirements:**
- `./ODBs/` from Step 2

**Command:**
```bash
cd c:\Users\lpatrign\Desktop\python_paper
python FCLGA_GraphTransformer/legacy/extractodb.py
```

**Expected Time:** < 1 minute

**Output:**
- `./ODBsONLY/` with all .odb files

---

### Step 5: Extract Results

**Script:** `fclga_extract_results.py`

**What it does:** Extracts E11 strain from .odb files

**Requirements:**
- Abaqus Python
- `./ODBsONLY/` from Step 4

**Command:**
```bash
cd c:\Users\lpatrign\Desktop\python_paper
abaqus cae nogui=FCLGA_GraphTransformer/src/preprocessing/fclga_extract_results.py
```

**Expected Time:** 20-60 minutes

**Output:**
- `./strains/` directory
- `E11_Plate_nonlinear_*.txt` files (500 files)

---

### Step 6: Build Dataset

**Script:** `fclga_build_dataset.py`

**What it does:** Combines strain files into single tensor

**Requirements:**
- Python with torch
- `./strains/` from Step 5

**Command:**
```bash
cd c:\Users\lpatrign\Desktop\python_paper
python FCLGA_GraphTransformer/src/preprocessing/fclga_build_dataset.py
```

**Expected Time:** < 1 minute

**Output:**
- `strains.pt` - Strain tensor

---

## 🧪 Testing

After each step, run:

```bash
cd FCLGA_GraphTransformer
python test_preprocessing.py
```

This shows which steps are complete.

---

## 🐛 Common Issues

### 1. Path Problems
Scripts expect to run from `c:\Users\lpatrign\Desktop\python_paper`

**Fix:** Always `cd` to this directory first

### 2. Abaqus Python Can't Find Torch
**Fix:** Install torch in Abaqus Python environment first

### 3. .inp Files Not Found
**Fix:** Update path in `fclga_extract_features.py`:
```python
inp_directory = "./INPs"  # Add this line
```

### 4. Too Slow
**Fix:** Test with smaller dataset:
- Change N=2, M=5 in geometry generation (10 samples)
- Reduce `num_parallel_simulations` in simulation script

---

## 📊 Estimated Total Time

- **Full Pipeline (500 samples):** 6-12 hours
- **Small Test (10 samples):** 30-60 minutes
- **Without Abaqus steps:** < 10 minutes (if you have existing data)

---

## ✅ Next Steps

**Recommended approach:**

1. **First:** Check if you have any existing data from previous runs
2. **Then:** Try small test (10 samples) to verify everything works
3. **Finally:** Run full pipeline (500 samples) overnight

**Which approach would you like to take?**
