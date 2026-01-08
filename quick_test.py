#!/usr/bin/env python3
"""
Quick Test Script - Auto-loads hyperparameters from training run

Usage:
    python quick_test.py

This will automatically load hyperparameters and model from:
    results/training_run_20260107_152157/

No need to specify any hyperparameters manually!
"""

import subprocess
import sys

training_run = "results/training_run_20260107_152157"

print("=" * 80)
print("RUNNING TEST WITH AUTO-LOADED HYPERPARAMETERS")
print("=" * 80)
print(f"Training run: {training_run}\n")

cmd = [
    sys.executable,
    "-m",
    "src.evaluation.fclga_test",
    "--training_run",
    training_run,
    "--visualize_samples",
    "5",
]

print("Command:")
print(" ".join(cmd))
print("\n" + "=" * 80 + "\n")

subprocess.run(cmd)
