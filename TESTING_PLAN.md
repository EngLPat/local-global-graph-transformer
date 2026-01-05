# Preprocessing Scripts Test Plan

## Testing Order

The preprocessing pipeline must run in this exact order:

1. **fclga_generate_geometry.py** - Creates .inp files
2. **fclga_run_simulations.py** - Runs Abaqus simulations → creates .odb files
3. **fclga_extract_features.py** - Reads .inp files → creates node_gnn_data.pt
4. **fclga_extract_results.py** - Reads .odb files → creates strain .txt files
5. **fclga_build_dataset.py** - Combines everything → creates strains.pt

---

## Test 1: fclga_generate_geometry.py

**Purpose:** Generate parametric plate geometries with holes and create Abaqus input files

**Requirements:**
- Abaqus Python environment
- Torch installed in Abaqus Python

**Expected Inputs:** None (generates data)

**Expected Outputs:**
- `./INPs/` directory with .inp files (500 files)
- `plate_geometry_data.pt` - PyTorch tensor with geometry parameters

**Command:**
```bash
abaqus cae nogui=src/preprocessing/fclga_generate_geometry.py
```

**What to Check:**
- [ ] INPs directory created
- [ ] 500 .inp files generated
- [ ] plate_geometry_data.pt file created
- [ ] No Python errors
- [ ] Files have reasonable sizes (not empty)

**Known Issues:**
- Requires Abaqus license
- Needs torch in Abaqus Python environment
- Takes significant time (~minutes to hours)

---

## Test 2: fclga_run_simulations.py

**Purpose:** Run Abaqus FEA simulations in parallel

**Requirements:**
- Abaqus
- ./INPs/ directory with .inp files from Test 1

**Expected Inputs:**
- `./INPs/*.inp` files

**Expected Outputs:**
- `./ODBs/` directory with subdirectories for each simulation
- Each subdirectory contains: .odb, .dat, .msg, .sta, .log, .com, .prt, .sim files

**Command:**
```bash
python src/preprocessing/fclga_run_simulations.py
```

**What to Check:**
- [ ] ODBs directory created
- [ ] Subdirectories for each model
- [ ] .odb files present in each subdirectory
- [ ] No Abaqus errors
- [ ] All simulations completed

**Configuration:**
- `num_parallel_simulations = 10` in script (line ~20)
- Adjust based on your CPU cores

**Known Issues:**
- Very time-consuming (hours)
- Requires Abaqus licenses (may need multiple for parallel)
- High CPU usage

---

## Test 3: fclga_extract_features.py

**Purpose:** Parse mesh geometry from .inp files and create graph structure

**Requirements:**
- Python environment with torch, torch_geometric
- ./INPs/*.inp files from Test 1

**Expected Inputs:**
- `./INPs/*.inp` files

**Expected Outputs:**
- `node_gnn_data.pt` - PyTorch Geometric Data objects
- `triangulation_data.pkl` - Triangulation for visualization

**Command:**
```bash
python src/preprocessing/fclga_extract_features.py
```

**What to Check:**
- [ ] node_gnn_data.pt created
- [ ] triangulation_data.pkl created
- [ ] No parsing errors
- [ ] Files have reasonable sizes
- [ ] Can load the .pt file: `torch.load('node_gnn_data.pt')`

**Known Issues:**
- Assumes .inp files are in current directory (not ./INPs/)
- May need to adjust file paths

---

## Test 4: fclga_extract_results.py

**Purpose:** Extract strain values from Abaqus .odb files

**Requirements:**
- Abaqus Python environment
- ./ODBsONLY/*.odb files (need to run extractodb.py first)

**Expected Inputs:**
- `./ODBsONLY/*.odb` files

**Expected Outputs:**
- `./strains/` directory
- `./strains/E11_Plate_nonlinear_*.txt` files (one per model)

**Command:**
```bash
# First, copy .odb files to ODBsONLY
python legacy/extractodb.py

# Then extract strains
abaqus cae nogui=src/preprocessing/fclga_extract_results.py
```

**What to Check:**
- [ ] strains/ directory created
- [ ] E11_*.txt files created (500 files)
- [ ] Each file contains node_id, strain_value pairs
- [ ] No Abaqus errors
- [ ] Values are reasonable (not all zeros, not NaN)

**Known Issues:**
- Requires Abaqus Python
- Hard-coded to extract E11 only
- Need to run extractodb.py first

---

## Test 5: fclga_build_dataset.py

**Purpose:** Combine strain data into single tensor

**Requirements:**
- Python with torch
- ./strains/E11_*.txt files from Test 4

**Expected Inputs:**
- `./strains/E11_Plate_nonlinear_*.txt` files

**Expected Outputs:**
- `strains.pt` - PyTorch tensor of shape [num_samples, max_nodes, 1]

**Command:**
```bash
python src/preprocessing/fclga_build_dataset.py
```

**What to Check:**
- [ ] strains.pt created
- [ ] Can load: `torch.load('strains.pt')`
- [ ] Check shape: should be [500, max_nodes, 1]
- [ ] Values are reasonable

**Known Issues:**
- Assumes ./strains/ directory exists
- Padding may add extra row (noted in code)

---

## Final Integration Test

After all individual tests pass, verify the complete dataset can be created:

**Requirements:**
- node_gnn_data.pt (features)
- strains.pt (targets)

**Expected Output:**
- Combined dataset for training

**Check:**
```python
import torch

# Load features and targets
features = torch.load('node_gnn_data.pt')
targets = torch.load('strains.pt')

print(f"Features: {len(features)} samples")
print(f"Targets shape: {targets.shape}")
print(f"Sample 0 nodes: {features[0].x.shape[0]}")
```

---

## Quick Test (Without Abaqus)

If you don't have Abaqus access or want to skip time-consuming steps:

1. **Test only fclga_extract_features.py** using existing .inp files
2. **Test only fclga_build_dataset.py** using existing strain .txt files
3. Check if outputs match legacy versions

---

## Test Status

- [ ] Test 1: fclga_generate_geometry.py
- [ ] Test 2: fclga_run_simulations.py
- [ ] Test 3: fclga_extract_features.py
- [ ] Test 4: fclga_extract_results.py
- [ ] Test 5: fclga_build_dataset.py
- [ ] Final Integration Test

---

## Notes

- All scripts currently run in current working directory
- May need to adjust paths to use config.PathConfig
- Test on small subset first (e.g., 10 samples instead of 500)
