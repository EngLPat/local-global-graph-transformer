# FCLGA GraphTransformer - Progress Tracker

## Authors & Publication

**Title:** Graph Neural Networks with Hybrid Local-Global Attention for Effective Prediction of Mechanical Response in Structures

**Authors:** Luca Patrignani, Silvestre T. Pinho  
**Institution:** Imperial College London  
**Journal:** Computer Methods in Applied Mechanics and Engineering

---

##  Phase 1: Structure Creation - COMPLETE

Created professional project structure with all folders, configuration files, and documentation.

**Completed:**
-  Created folder structure
-  Created `.gitignore`, `requirements.txt`, `README.md`, `LICENSE`
-  Backed up 9 original files to `legacy/`
-  Created all `__init__.py` files
-  Created `config/defaults.yaml`

---

##  Phase 2A: File Copying - COMPLETE

Copied all files from `legacy/` to new structure with proper names. **NO CODE MODIFICATIONS YET.**

### Preprocessing Files (src/preprocessing/)
-  `fclga_generate_geometry.py`  Create_Composite_final_nodes_nonlinear.py
-  `fclga_run_simulations.py`  RUN_Plate_Parallel.py
-  `fclga_extract_features.py`  openhole_features_composite_nodes.py
-  `fclga_extract_results.py`  openhole_E_nodes_11.py
-  `fclga_build_dataset.py`  strains_tensor_nonlinear.py

### Training Files (src/training/)
-  `fclga_train_model.py`  GNN_MeshGraphNets_Train_Test_Luca_2_with_frequency.py

### Evaluation Files (src/evaluation/)
-  `fclga_test_model.py`  Testing_MeshGraphNets_2_optmized_svg_separated_NEWRMSE_newattention.py

### Entry Point Scripts (scripts/)
-  `fclga_run_pipeline.py` - Main orchestrator (placeholder)
-  `fclga_train.py` - Training entry point (placeholder)
-  `fclga_test.py` - Testing entry point (placeholder)

---

##  Current Structure

```
FCLGA_GraphTransformer/
 src/
    preprocessing/           5 files copied
    models/                  Empty (Phase 2B)
    training/                1 file copied
    evaluation/              1 file copied
    utils/                   Empty (Phase 2C)
 scripts/                     3 placeholder scripts
 legacy/                      9 original files (SAFE)
 config/                      Configuration files
 tests/                       Empty (Phase 3)
 [docs, requirements, etc.]   Complete
```

---

##  Next Steps: Phase 2B

### Phase 2B: Extract Model Components (NO logic changes)

Will extract model classes from training script to separate files:

1. **Extract to `src/models/fclga_transformer.py`:**
   - `FCLGA_GraphTransformer` class (rename from MeshGraphNet)
   - Model initialization and forward pass

2. **Extract to `src/models/layers.py`:**
   - `ProcessorLayer` class
   - `GlobalAttention` class

3. **Extract to `src/utils/normalization.py`:**
   - `normalize()` function
   - `unnormalize()` function

4. **Extract to `src/utils/io_utils.py`:**
   - File I/O helper functions
   - Path management utilities

### Phase 2C: Add Documentation (Safe changes)

- Add brief docstrings to main functions
- Add type hints to function signatures
- Add module-level documentation
- NO logic changes, only documentation

### Phase 2D: Code Quality

- Run `black` for formatting
- Run `isort` for import sorting
- Run `flake8` for basic linting
- Fix obvious code style issues

---

##  Important Notes

**Risk Level:** ZERO - No code has been modified yet!

- Original working code is safely in `legacy/` folder
- All copied files are identical to originals
- Can compare at any time using file diff tools
- Can revert to legacy at any moment

**Testing Strategy:**

After each phase:
1. Compare outputs with `legacy/` code
2. Verify behavior is identical
3. Commit to git if successful
4. Proceed to next phase

---

##  Current Status

**Phase 2A: File Copying**  COMPLETE

Ready to proceed to Phase 2B: Extract Model Components

---

##  How to Use Current Structure

### Run Original Code (Still Works!)
```bash
cd legacy
python GNN_MeshGraphNets_Train_Test_Luca_2_with_frequency.py
```

### Run New Structure (Same code, new names)
```bash
# Preprocessing
python src/preprocessing/fclga_generate_geometry.py
python src/preprocessing/fclga_run_simulations.py
python src/preprocessing/fclga_extract_features.py
python src/preprocessing/fclga_extract_results.py
python src/preprocessing/fclga_build_dataset.py

# Training
python src/training/fclga_train_model.py

# Testing
python src/evaluation/fclga_test_model.py

# Pipeline (placeholder for now)
python scripts/fclga_run_pipeline.py --stage all
```

---

**Questions? Ready for Phase 2B?**
