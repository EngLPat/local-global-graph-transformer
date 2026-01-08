"""
FCLGA GraphTransformer - Testing Entry Point

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London

Standalone testing script.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.fclga_test_model import *


if __name__ == '__main__':
    print("FCLGA GraphTransformer - Testing")
    print("=" * 60)
    print("\nFor now, run the testing script directly:")
    print("python -m src.evaluation.fclga_test_model")
    print("\n(Standalone entry point coming in Phase 3)")
