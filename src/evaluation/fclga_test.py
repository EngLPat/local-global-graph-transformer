"""
FCLGA GraphTransformer - Model Testing Script

Command-line interface for testing trained FCLGA GraphTransformer models.
Supports loading legacy model weights and evaluating on test datasets.

Usage:
    python -m src.evaluation.fclga_test \
        --model_path path/to/model.pt \
        --dataset_path path/to/dataset.pt \
        --num_layers 5 \
        --hidden_dim 128

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import torch
import random
import numpy as np
import argparse
import os
import json
from pathlib import Path

from src.models.fclga_graph_transformer import FCLGA_GraphTransformer
from src.utils.data_utils import get_stats
from src.utils.test_utils import (
    ObjectView,
    visualize_sample,
    evaluate_model,
    benchmark_inference
)
from config import paths as config_paths


def load_hyperparameters_from_training(training_run_path):
    """
    Load hyperparameters and model path from training run directory.
    
    Args:
        training_run_path (str): Path to training run directory.
        
    Returns:
        tuple: (hyperparameters_dict, model_path_str)
    """
    run_path = Path(training_run_path)
    
    if not run_path.exists():
        raise FileNotFoundError(f"Training run directory not found: {run_path}")
    
    # Find hyperparameters JSON
    hyp_json = run_path / 'hyperparameter_optimization_results' / 'best_hyperparameters.json'
    if not hyp_json.exists():
        raise FileNotFoundError(
            f"Hyperparameters file not found: {hyp_json}\n"
            f"Make sure the training run completed with Optuna optimization."
        )
    
    # Load hyperparameters
    print(f"Loading hyperparameters from: {hyp_json}")
    with open(hyp_json, 'r') as f:
        params = json.load(f)
    
    # Find best model
    best_models_dir = run_path / 'best_models'
    if not best_models_dir.exists():
        raise FileNotFoundError(f"Best models directory not found: {best_models_dir}")
    
    model_files = list(best_models_dir.glob('*.pt'))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {best_models_dir}")
    
    # Prefer FINAL model if it exists, otherwise use most recent
    final_models = [m for m in model_files if 'FINAL' in m.name]
    if final_models:
        model_path = final_models[0]
        print(f"Found final trained model: {model_path.name}")
    else:
        # Use newest model by modification time
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"Found best model: {model_path.name}")
    
    print("\nLoaded hyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    return params, str(model_path)


def parse_arguments():
    """
    Parse command-line arguments for model testing.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Test FCLGA GraphTransformer model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data paths
    parser.add_argument(
        '--training_run',
        type=str,
        default=None,
        help='Path to training run directory (auto-loads hyperparameters and model). '
             'If provided, overrides --model_path and hyperparameter arguments.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to trained model checkpoint (.pt file). '
             'Required if --training_run is not provided.'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Path to dataset (.pt file). If not provided, uses data/processed/{material_type}/datasets/processed_data.pt'
    )
    parser.add_argument(
        '--material_type',
        type=str,
        default='nonlinear',
        choices=['linear', 'nonlinear'],
        help='Material type: linear (elastic) or nonlinear (plastic) - default: nonlinear'
    )

    # Model hyperparameters (must match training configuration)
    parser.add_argument(
        '--num_layers',
        type=int,
        default=5,
        help='Number of message passing layers'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='Hidden dimension size'
    )
    parser.add_argument(
        '--attention_freq',
        type=int,
        default=4,
        help='Global attention frequency (apply every N layers)'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.21887774707222715,
        help='Dropout rate'
    )

    # Dataset parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )

    # Testing options
    parser.add_argument(
        '--num_benchmark_runs',
        type=int,
        default=100,
        help='Number of runs for inference benchmarking'
    )
    parser.add_argument(
        '--visualize_samples',
        type=int,
        default=25,
        help='Number of test samples to visualize'
    )

    args = parser.parse_args()
    
    # Validate arguments and auto-load if training_run is provided
    if args.training_run is not None:
        # Auto-load mode: load hyperparameters and model from training run
        print("="*80)
        print("AUTO-LOADING FROM TRAINING RUN")
        print("="*80)
        print(f"Training run directory: {args.training_run}\n")
        
        try:
            params, model_path = load_hyperparameters_from_training(args.training_run)
            
            # Override args with loaded hyperparameters
            args.model_path = model_path
            args.num_layers = params['num_layers']
            args.hidden_dim = params['hidden_dim']
            args.dropout_rate = params['dropout_rate']
            args.attention_freq = params['attention_freq']
            args.batch_size = params['batch_size']
            
            # Note: train_size no longer needed - using percentage-based splitting
            
            print("\n✓ Hyperparameters and model path loaded successfully!")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n✗ Error loading from training run: {e}")
            print("\nPlease ensure:")
            print("  1. The training run directory exists")
            print("  2. It contains hyperparameter_optimization_results/best_hyperparameters.json")
            print("  3. It contains best_models/*.pt files")
            raise
    
    elif args.model_path is None:
        # Neither training_run nor model_path provided
        parser.error("Either --training_run or --model_path must be provided")
    
    return args


def load_and_split_dataset(dataset_path, shuffle=True, seed=5):
    """
    Load dataset and split into train/val/test sets using percentage-based splitting.
    
    Splits dataset as 70% train / 15% val / 15% test to match legacy behavior.
    This ensures consistent splitting regardless of dataset size:
    - 500 samples → 350/75/75 (matches legacy)
    - 700 samples → 490/105/105
    - Any size scales proportionally

    Args:
        dataset_path (str): Path to dataset file.
        shuffle (bool, optional): Whether to shuffle before splitting. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to 5.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, full_dataset)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")
    dataset = torch.load(dataset_path, weights_only=False)

    # Shuffle dataset before splitting (matches legacy training behavior)
    if shuffle:
        print(
            f"Shuffling dataset with seed={seed} "
            f"(to match legacy training)..."
        )
        random.shuffle(dataset)

    # Split dataset using percentages (70% train, 15% val, 15% test)
    total_size = len(dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    print("\nDataset split (70%/15%/15%):")
    print(f"  Total size: {total_size}")
    print(f"  Training: {train_size} (70%)")
    print(f"  Validation: {val_size} (15%)")
    print(f"  Test: {test_size} (15%)")

    return train_dataset, val_dataset, test_dataset, dataset


def create_model(num_node_features, num_edge_features, args, device):
    """
    Create and initialize FCLGA_GraphTransformer model.

    Args:
        num_node_features (int): Number of input node features.
        num_edge_features (int): Number of input edge features.
        args: Arguments object with model hyperparameters.
        device (str): Device to place model on ('cuda' or 'cpu').

    Returns:
        FCLGA_GraphTransformer: Initialized model on specified device.
    """
    num_classes = 1  # Strain prediction (single output)

    print("\nCreating model architecture...")
    print(f"  Node features: {num_node_features}")
    print(f"  Edge features: {num_edge_features}")
    print(f"  Output dimension: {num_classes}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Attention frequency: {args.attention_freq}")

    model = FCLGA_GraphTransformer(
        num_node_features,
        num_edge_features,
        args.hidden_dim,
        num_classes,
        args
    ).to(device)

    return model


def load_model_weights(model, model_path, device):
    """
    Load trained weights into model.

    Args:
        model: FCLGA_GraphTransformer model to load weights into.
        model_path (str): Path to model checkpoint file.
        device (str): Device for loading ('cuda' or 'cpu').

    Raises:
        FileNotFoundError: If model checkpoint file doesn't exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"\nLoading model weights from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle legacy checkpoints with skip_projection or other mismatched keys
    model_state = model.state_dict()
    filtered_checkpoint = {}
    
    for key, value in checkpoint.items():
        if key in model_state:
            # Check if shapes match
            if model_state[key].shape == value.shape:
                filtered_checkpoint[key] = value
            else:
                print(f"⚠ Skipping {key}: shape mismatch (checkpoint: {value.shape}, model: {model_state[key].shape})")
        else:
            print(f"⚠ Skipping {key}: not in current model architecture")
    
    # Load filtered state dict (strict=False allows missing keys)
    model.load_state_dict(filtered_checkpoint, strict=False)
    model.eval()
    print("✓ Model weights loaded successfully!")


def main():
    """Main testing workflow."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set random seeds for reproducibility (matches legacy behavior)
    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)

    # Setup directories
    # If training_run was provided, put test results in that directory
    if args.training_run is not None:
        postprocess_dir = os.path.join(args.training_run, 'test_results')
        os.makedirs(postprocess_dir, exist_ok=True)
    else:
        # Only create TEST_RESULTS_DIR when not using training_run
        postprocess_dir = str(config_paths.TEST_RESULTS_DIR)
        os.makedirs(postprocess_dir, exist_ok=True)
    
    checkpoint_dir = str(config_paths.BEST_MODELS_DIR)

    # Create args object for model (legacy compatibility)
    model_params = {
        'model_type': 'meshgraphnet',
        'num_layers': args.num_layers,
        'attention_freq': args.attention_freq,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'dropout_rate': args.dropout_rate,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'shuffle': True,
        'checkpoint_dir': checkpoint_dir,
        'postprocess_dir': postprocess_dir
    }
    model_args = ObjectView(model_params)

    # Print header
    print("=" * 70)
    print("FCLGA GRAPHTRANSFORMER - MODEL TESTING")
    print("=" * 70)
    print(f"\nModel checkpoint: {args.model_path}")
    print(f"Device: {model_args.device}")

    # Load and split dataset
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        from config.paths import get_paths
        paths = get_paths(args.material_type)
        dataset_path = str(paths.DATASETS_DIR / 'processed_data.pt')
    train_dataset, val_dataset, test_dataset, full_dataset = (
        load_and_split_dataset(dataset_path)
    )

    # Get normalization statistics from training data only (matches legacy, prevents data leakage)
    stats_list = get_stats(train_dataset)

    # Create model
    device = model_args.device
    num_node_features = test_dataset[0].x.shape[1]
    num_edge_features = test_dataset[0].edge_attr.shape[1]

    model = create_model(num_node_features, num_edge_features, model_args, device)

    # Load trained weights
    load_model_weights(model, args.model_path, device)

    # Step 1: Benchmark inference time
    print("\n" + "=" * 70)
    print("STEP 1: BENCHMARKING INFERENCE TIME")
    print("=" * 70)
    benchmark_results = benchmark_inference(
        model,
        test_dataset,
        device,
        stats_list,
        postprocess_dir,
        num_runs=args.num_benchmark_runs
    )

    # Step 2: Evaluate on test set
    print("\n" + "=" * 70)
    print("STEP 2: EVALUATING ON TEST SET")
    print("=" * 70)
    final_test_loss, final_test_rmse = evaluate_model(
        test_dataset,
        model,
        device,
        stats_list,
        postprocess_dir
    )

    # Print final summary
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Test Loss: {final_test_loss:.5f}")
    print(f"Test RMSE: {final_test_rmse:.8f}")
    print(f"Inference Time: {benchmark_results['avg_inference_time'] * 1000:.3f} ms")
    print(f"Speedup vs FEM: {benchmark_results['speedup']:.0f}×")
    print(f"{'=' * 70}\n")

    # Step 3: Generate visualizations
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 70)
    num_visualize = min(args.visualize_samples, len(test_dataset))
    print(f"Generating {num_visualize} visualizations...")

    for i in range(num_visualize):
        plot_name = f'test_sample_{i}_results'
        visualize_sample(model, test_dataset, i, postprocess_dir, plot_name, stats_list)

        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{num_visualize} visualizations")

    print(f"\n✓ All visualizations saved to: {postprocess_dir}")
    print("\n" + "=" * 70)
    print("TESTING COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
