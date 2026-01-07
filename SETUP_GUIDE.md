# Setup Guide for New Contributors

## 🎯 Complete Setup in 4 Steps

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Clone Repository                                  │
├─────────────────────────────────────────────────────────────┤
│  $ git clone https://github.com/user/FCLGA_GraphTransformer│
│  $ cd FCLGA_GraphTransformer                                │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Create Conda Environment                          │
├─────────────────────────────────────────────────────────────┤
│  $ conda env create -f environment.yml                      │
│  ⏱️  Takes ~5-10 minutes                                    │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Activate Environment                               │
├─────────────────────────────────────────────────────────────┤
│  $ conda activate fclga                                     │
│  ✓ Check: prompt shows (fclga)                              │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Verify Installation                                │
├─────────────────────────────────────────────────────────────┤
│  $ python test_environment.py                               │
│  ✓ All packages should be available                         │
└─────────────────────────────────────────────────────────────┘
```

## 📝 Environment Activation Checklist

**Before running ANY Python script, always check:**

```bash
# ❌ BAD - Will fail with "ModuleNotFoundError"
(base) user@computer:~/project$ python scripts/validate_processed_data.py

# ✅ GOOD - Will work correctly  
(fclga) user@computer:~/project$ python scripts/validate_processed_data.py
         ^^^^^^
         You should see this!
```

## 🔄 Daily Workflow

```
Morning:
  1. cd /path/to/FCLGA_GraphTransformer
  2. conda activate fclga          ← CRITICAL STEP
  3. [work on your code/run scripts]

Evening:
  4. conda deactivate              ← Clean exit
```

## 🆘 Most Common Mistake

### Problem
```bash
$ python scripts/validate_processed_data.py
Traceback (most recent call last):
  File "scripts/validate_processed_data.py", line 3, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

### Solution
```bash
$ conda activate fclga
(fclga) $ python scripts/validate_processed_data.py
# Now it works! ✓
```

## 🎓 Understanding Conda Environments

Think of conda environments like **isolated workspaces**:

```
Your Computer
├── base environment        ← Default, minimal packages
├── fclga environment      ← Our project, all packages installed here ✓
├── other_project          ← Some other project
└── ...

When you run: conda activate fclga
You're saying: "Use the fclga workspace with all its packages"
```

## 🔍 Environment Commands Cheat Sheet

```bash
# List all environments (active one has *)
conda env list

# Activate an environment
conda activate fclga

# Deactivate current environment
conda deactivate

# Update environment from changed environment.yml
conda env update -f environment.yml --prune

# Remove an environment completely
conda env remove -n fclga

# Export your current environment
conda env export > my_environment.yml
```

## 📂 What Gets Installed?

When you run `conda env create -f environment.yml`, you get:

```
Python 3.10
├── PyTorch (GPU/CPU)
├── PyTorch Geometric (for Graph Neural Networks)
├── NumPy (numerical computing)
├── SciPy (scientific computing)
├── Pandas (data manipulation)
├── Matplotlib (plotting)
├── Seaborn (statistical visualization)
├── PyYAML (configuration files)
├── tqdm (progress bars)
└── pytest (testing)
```

Total size: ~2-3 GB

## 🔧 Advanced: Auto-Activation

Add this to `~/.bashrc` (Linux/Mac) or `~/.bash_profile`:

```bash
# Auto-activate fclga when entering project directory
function cd() {
    builtin cd "$@"
    if [[ -f environment.yml ]] && [[ $(basename "$PWD") == "FCLGA_GraphTransformer" ]]; then
        if [[ "$CONDA_DEFAULT_ENV" != "fclga" ]]; then
            echo "Auto-activating fclga environment..."
            conda activate fclga
        fi
    fi
}
```

Then:
```bash
$ source ~/.bashrc
$ cd FCLGA_GraphTransformer
Auto-activating fclga environment...
(fclga) $ # Already activated! ✓
```

## 🌐 Multiple Machines Setup

Working on multiple computers? Here's the workflow:

### Machine 1 (Initial Setup)
```bash
git clone https://github.com/user/FCLGA_GraphTransformer.git
cd FCLGA_GraphTransformer
conda env create -f environment.yml
conda activate fclga
```

### Machine 2 (Same Steps!)
```bash
git clone https://github.com/user/FCLGA_GraphTransformer.git
cd FCLGA_GraphTransformer
conda env create -f environment.yml
conda activate fclga
```

The `environment.yml` file ensures **identical setups** on all machines!

## 📚 Further Reading

- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

## ✅ Quick Test Script

Save this as `test_environment.py` (already in the repo):

```python
#!/usr/bin/env python
"""Test if the environment is properly configured."""

import sys

def test_imports():
    """Test all critical imports."""
    tests = {
        'Python': lambda: sys.version_info >= (3, 8),
        'NumPy': lambda: __import__('numpy'),
        'PyTorch': lambda: __import__('torch'),
        'PyTorch Geometric': lambda: __import__('torch_geometric'),
        'Matplotlib': lambda: __import__('matplotlib'),
        'SciPy': lambda: __import__('scipy'),
    }
    
    print("Testing environment setup...\n")
    all_passed = True
    
    for name, test_func in tests.items():
        try:
            result = test_func()
            print(f"✓ {name}: OK")
        except Exception as e:
            print(f"✗ {name}: FAILED - {e}")
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All tests passed! Environment is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check your installation.")
        return 1

if __name__ == '__main__':
    sys.exit(test_imports())
```

Run with:
```bash
conda activate fclga
python test_environment.py
```

---

**Remember: The #1 rule is always activate the environment first!** 🚀
