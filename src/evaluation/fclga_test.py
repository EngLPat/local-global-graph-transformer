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

from src.models.fclga_graph_transformer import FCLGA_GraphTransformer
from src.utils.data_utils import get_stats
from src.utils.test_utils import (
    ObjectView,
    visualize_sample,
    evaluate_model,
    benchmark_inference
)
from config import paths as config_paths


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
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Path to dataset (.pt file). If not provided, uses datasets/processed_data.pt'
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
        '--train_size',
        type=int,
        default=400,
        help='Training set size (for dataset splitting)'
    )
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

    return parser.parse_args()


def load_and_split_dataset(dataset_path, train_size, shuffle=True, seed=5):
    """
    Load dataset and split into train/val/test sets.

    Args:
        dataset_path (str): Path to dataset file.
        train_size (int): Number of samples for training set.
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

    # Split dataset
    val_size = 1
    test_size = len(dataset) - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    print("\nDataset split:")
    print(f"  Total size: {len(dataset)}")
    print(f"  Training: {train_size}")
    print(f"  Validation: {val_size}")
    print(f"  Test: {test_size}")

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
    config_paths.setup_directories()
    checkpoint_dir = str(config_paths.BEST_MODELS_DIR)
    postprocess_dir = str(config_paths.PLOTS_DIR)

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
        dataset_path = str(config_paths.DATASETS_DIR / 'processed_data.pt')
    train_dataset, val_dataset, test_dataset, full_dataset = (
        load_and_split_dataset(dataset_path, args.train_size)
    )

    # Get normalization statistics from full dataset
    stats_list = get_stats(full_dataset)

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
