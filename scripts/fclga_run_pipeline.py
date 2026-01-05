"""
FCLGA GraphTransformer - Main Pipeline Script

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
Paper: "Graph Neural Networks with Hybrid Local-Global Attention for 
        Effective Prediction of Mechanical Response in Structures"
Journal: Computer Methods in Applied Mechanics and Engineering

This script orchestrates the complete pipeline from geometry generation to model testing.
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def main():
    parser = argparse.ArgumentParser(
        description='FCLGA GraphTransformer Pipeline'
    )
    parser.add_argument(
        '--stage',
        choices=['all', 'preprocess', 'train', 'test'],
        default='all',
        help='Pipeline stage to run'
    )
    parser.add_argument(
        '--num_cpus',
        type=int,
        default=None,
        help='Number of CPUs for parallel simulation (default: auto-detect)'
    )
    parser.add_argument(
        '--output_variables',
        nargs='+',
        default=['E11'],
        help='Variables to extract from FEA results'
    )
    
    args = parser.parse_args()
    
    # Auto-detect CPUs if not specified
    if args.num_cpus is None:
        import multiprocessing
        args.num_cpus = max(1, multiprocessing.cpu_count() // 2)
        print(f"Auto-detected CPUs: {args.num_cpus}")
    
    print(f"\n{'='*60}")
    print(f"FCLGA GraphTransformer Pipeline")
    print(f"{'='*60}\n")
    
    if args.stage in ['all', 'preprocess']:
        print("Stage 1: Preprocessing")
        print("-" * 60)
        print("1. Generating geometry...")
        print("2. Running FEA simulations...")
        print("3. Extracting features...")
        print("4. Extracting results...")
        print("5. Building dataset...")
        print("\nRun individual preprocessing scripts for now.")
        print("(Integration coming in Phase 3)\n")
    
    if args.stage in ['all', 'train']:
        print("Stage 2: Training")
        print("-" * 60)
        print("Run: python src/training/fclga_train_model.py")
        print("(Main entry point coming in Phase 3)\n")
    
    if args.stage in ['all', 'test']:
        print("Stage 3: Testing")
        print("-" * 60)
        print("Run: python src/evaluation/fclga_test_model.py")
        print("(Main entry point coming in Phase 3)\n")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
