"""
FCLGA GraphTransformer - Training Entry Point

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London

Standalone training script.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.fclga_train_model import *


if __name__ == '__main__':
    print("FCLGA GraphTransformer - Training")
    print("=" * 60)
    print("\nFor now, run the training script directly:")
    print("python src/training/fclga_train_model.py")
    print("\n(Standalone entry point coming in Phase 3)")
