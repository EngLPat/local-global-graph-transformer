"""
*******************************************************************************
*                                                                             *
*   AUTHOR: Luca Patrignani - PhD candidate Imperial College London           *
*   TITLE: GNN MeshGraphNets Testing with Legacy Weights                      *
*   DATE: 06/01/2026                                                          *
*                                                                             *
*******************************************************************************
*                                                                             *
*  Description:                                                               *
*  ============                                                               *
*  Standalone testing script for loading legacy trained model weights         *
*  and evaluating them using the refactored codebase.                         *
*  Preserves all functionality from the original testing script.              *
*                                                                             *
*  Usage:                                                                     *
*  =====                                                                      *
*  python src/evaluation/test_legacy_weights.py \                             *
*      --model_path results/best_models/model.pt \                            *
*      --num_layers 5 --hidden_dim 128 --attention_freq 4 \                   *
*      --train_size 400 --test_size 100                                       *
*                                                                             *
*******************************************************************************
"""

import torch
import random
import numpy as np
import argparse
import os
import time
from torch_geometric.loader import DataLoader

# Import refactored modules
from src.models.fclga_graph_transformer import FCLGA_GraphTransformer
from src.utils.data_utils import normalize, unnormalize, get_stats
from src.utils.visualization import plot_results

# Import configuration
from config import paths as config_paths


class objectview(object):
    """Simple class to convert dictionary to object with attributes."""
    def __init__(self, d):
        self.__dict__ = d


def visualize(loader_original, model, file_dir, plot_name, stats_list, sample_index=0):
    """
    Visualize model predictions on a single sample.
    Preserves legacy functionality exactly.
    """
    model.eval()
    device = next(model.parameters()).device
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    
    # Move statistics to device
    mean_vec_x, std_vec_x = mean_vec_x.to(device), std_vec_x.to(device)
    mean_vec_edge, std_vec_edge = mean_vec_edge.to(device), std_vec_edge.to(device)
    mean_vec_y, std_vec_y = mean_vec_y.to(device), std_vec_y.to(device)
    
    # Create a new loader with batch size 1 just for visualization
    if hasattr(loader_original.dataset, '__getitem__'):
        # Get the specific sample we want to visualize
        single_sample = loader_original.dataset[sample_index]
        single_sample = single_sample.to(device)
        
        with torch.no_grad():
            pred = model(single_sample, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            pred = unnormalize(pred, mean_vec_y, std_vec_y)
        
        plot_results(single_sample, pred, file_dir, plot_name)


def evaluate_final_model(test_dataset, best_model, device, stats_list, args):
    """
    Evaluate model on test set with detailed RMSE metrics.
    Preserves legacy functionality exactly.
    """
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    
    # Move statistics to device
    mean_vec_x, std_vec_x = mean_vec_x.to(device), std_vec_x.to(device)
    mean_vec_edge, std_vec_edge = mean_vec_edge.to(device), std_vec_edge.to(device)
    mean_vec_y, std_vec_y = mean_vec_y.to(device), std_vec_y.to(device)
    
    best_model.eval()
    
    total_loss = 0
    num_samples = 0
    individual_rmses = []
    
    # Store all predictions and ground truths for global RMSE
    all_preds = []
    all_gts = []
    
    print("\nEvaluating on test set...")
    print("="*70)
    
    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            
            # Forward pass
            out = best_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            
            # Calculate loss
            loss = best_model.loss(out, data, mean_vec_y, std_vec_y)
            total_loss += loss.item()
            
            # Unnormalize for RMSE calculation
            eval_y = unnormalize(out, mean_vec_y, std_vec_y)
            gs_y = data.y
            
            # Store for global RMSE
            all_preds.append(eval_y.cpu().numpy())
            all_gts.append(gs_y.cpu().numpy())
            
            # Calculate per-sample RMSE with masking for padded nodes
            threshold = 1e-6
            non_zero_mask = torch.abs(gs_y) > threshold
            
            if non_zero_mask.sum() > 0:
                filtered_eval_y = eval_y[non_zero_mask]
                filtered_gs_y = gs_y[non_zero_mask]
                sample_rmse = torch.sqrt(torch.mean((filtered_eval_y - filtered_gs_y) ** 2)).item()
                individual_rmses.append(sample_rmse)
                
                # Print per-sample RMSE
                print(f"Sample {num_samples:3d}: RMSE = {sample_rmse:.8f} "
                      f"({non_zero_mask.sum().item():5d} non-zero nodes, "
                      f"{(~non_zero_mask).sum().item():4d} masked)")
            else:
                print(f"Sample {num_samples:3d}: No non-zero nodes found (all masked)")
                individual_rmses.append(0.0)
            
            num_samples += 1
    
    # Convert to numpy
    individual_rmses = np.array(individual_rmses)
    
    # Calculate GLOBAL RMSE across all nodes from all samples
    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    
    # Apply same masking to global data
    threshold = 1e-6
    non_zero_mask_global = np.abs(all_gts) > threshold
    filtered_all_preds = all_preds[non_zero_mask_global]
    filtered_all_gts = all_gts[non_zero_mask_global]
    
    global_rmse = np.sqrt(np.mean((filtered_all_preds - filtered_all_gts) ** 2))
    
    # Calculate mean of per-sample RMSEs
    mean_sample_rmse = np.mean(individual_rmses)
    
    # Calculate statistics
    mean_loss = total_loss / num_samples
    rmse_std = np.std(individual_rmses)
    rmse_min = np.min(individual_rmses)
    rmse_max = np.max(individual_rmses)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Number of test samples: {num_samples}")
    print(f"Mean Test Loss: {mean_loss:.5f}")
    print(f"\n--- RMSE METRICS ---")
    print(f"Global RMSE (all nodes, all samples): {global_rmse:.8f}")
    print(f"Mean of per-sample RMSEs: {mean_sample_rmse:.8f}")
    print(f"RMSE Std Dev: {rmse_std:.8f}")
    print(f"RMSE Range: [{rmse_min:.8f}, {rmse_max:.8f}]")
    print(f"\n--- MASKING INFO ---")
    print(f"Total nodes across all samples: {len(all_gts)}")
    print(f"Non-zero nodes (used): {non_zero_mask_global.sum()}")
    print(f"Masked nodes (padded/zero): {(~non_zero_mask_global).sum()}")
    print(f"Masking percentage: {((~non_zero_mask_global).sum() / len(all_gts) * 100):.2f}%")
    print(f"{'='*70}\n")
    
    # Save detailed results
    results_path = os.path.join(args.postprocess_dir, 'test_results_detailed.txt')
    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Number of test samples: {num_samples}\n")
        f.write(f"Mean Test Loss: {mean_loss:.5f}\n\n")
        f.write("--- RMSE METRICS ---\n")
        f.write(f"Global RMSE (all nodes, all samples): {global_rmse:.8f}\n")
        f.write(f"Mean of per-sample RMSEs: {mean_sample_rmse:.8f}\n")
        f.write(f"RMSE Std Dev: {rmse_std:.8f}\n")
        f.write(f"RMSE Range: [{rmse_min:.8f}, {rmse_max:.8f}]\n\n")
        f.write("--- MASKING INFO ---\n")
        f.write(f"Total nodes across all samples: {len(all_gts)}\n")
        f.write(f"Non-zero nodes (used): {non_zero_mask_global.sum()}\n")
        f.write(f"Masked nodes (padded/zero): {(~non_zero_mask_global).sum()}\n")
        f.write(f"Masking percentage: {((~non_zero_mask_global).sum() / len(all_gts) * 100):.2f}%\n\n")
        f.write("Individual Sample RMSEs:\n")
        for i, rmse in enumerate(individual_rmses):
            f.write(f"  Sample {i}: {rmse:.8f}\n")
    
    print(f"Detailed results saved to: {results_path}")
    
    # Visualize a representative sample (closest to mean RMSE)
    closest_idx = np.argmin(np.abs(individual_rmses - mean_sample_rmse))
    plot_name = f'test_representative_sample_{closest_idx}'
    single_loader = DataLoader([test_dataset[closest_idx]], batch_size=1, shuffle=False)
    visualize(single_loader, best_model, args.postprocess_dir, plot_name, stats_list)
    
    return mean_loss, global_rmse


def benchmark_inference_time(model, test_dataset, device, stats_list, num_runs=100):
    """
    Benchmark inference time and compare with FEM.
    Preserves legacy functionality exactly.
    """
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    
    # Move statistics to device
    mean_vec_x, std_vec_x = mean_vec_x.to(device), std_vec_x.to(device)
    mean_vec_edge, std_vec_edge = mean_vec_edge.to(device), std_vec_edge.to(device)
    mean_vec_y, std_vec_y = mean_vec_y.to(device), std_vec_y.to(device)
    
    model.eval()
    
    print("\n" + "="*60)
    print("INFERENCE TIME BENCHMARK")
    print("="*60 + "\n")
    
    # Use first test sample for benchmarking
    test_sample = test_dataset[0].to(device)
    
    # Warmup runs (to ensure GPU is ready)
    print("Performing warmup runs...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_sample, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Actual timing runs
    print(f"Running {num_runs} inference iterations...")
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(test_sample, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
        
        if device == 'cuda':
            torch.cuda.synchronize()
    
    end_time = time.time()
    total_inference_time = end_time - start_time
    avg_inference_time = total_inference_time / num_runs
    
    # FEM baseline (typical Abaqus simulation time for this problem)
    actual_fem_time = 120.0  # seconds, based on typical FEM simulation
    
    # Calculate speedup
    speedup = actual_fem_time / avg_inference_time
    time_saved = actual_fem_time - avg_inference_time
    
    results = {
        'avg_inference_time': avg_inference_time,
        'total_inference_time': total_inference_time,
        'num_runs': num_runs,
        'speedup': speedup,
        'time_saved': time_saved,
        'fem_time': actual_fem_time
    }
    
    print(f"\nGNN Inference Results:")
    print(f"  Average inference time: {avg_inference_time*1000:.3f} ms ({avg_inference_time:.6f} s)")
    print(f"  Number of runs: {num_runs}")
    print(f"  Total time: {total_inference_time:.3f} seconds")
    print(f"  Inference frequency: {1/avg_inference_time:.1f} predictions/second")
    
    print(f"\nSpeedup Analysis vs FEM:")
    print(f"  Abaqus FEM time: {actual_fem_time:.1f} seconds")
    print(f"  GNN inference time: {avg_inference_time*1000:.3f} ms")
    print(f"  Speedup: {results['speedup']:.0f}× faster than FEM")
    print(f"  Time saved per prediction: {results['time_saved']:.3f} seconds")
    
    # Break-even analysis for training cost
    training_time_estimate = 3600  # 1 hour estimate
    time_saved_per_pred = results['time_saved']
    breakeven_predictions = training_time_estimate / time_saved_per_pred if time_saved_per_pred > 0 else float('inf')
    
    print(f"\nBreak-even Analysis:")
    print(f"  Training time estimate: {training_time_estimate/3600:.1f} hours")
    print(f"  Break-even point: {breakeven_predictions:.0f} predictions")
    print(f"  Training ROI: Profitable after {breakeven_predictions:.0f} simulations")
    
    # Save results to file
    postprocess_dir = str(config_paths.PLOTS_DIR)
    benchmark_path = os.path.join(postprocess_dir, 'inference_benchmark.txt')
    with open(benchmark_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("INFERENCE TIME BENCHMARK\n")
        f.write("="*60 + "\n\n")
        f.write(f"GNN Inference Results:\n")
        f.write(f"  Average inference time: {avg_inference_time*1000:.3f} ms ({avg_inference_time:.6f} s)\n")
        f.write(f"  Number of runs: {num_runs}\n")
        f.write(f"  Total time: {total_inference_time:.3f} seconds\n")
        f.write(f"  Inference frequency: {1/avg_inference_time:.1f} predictions/second\n\n")
        f.write(f"Speedup Analysis vs FEM:\n")
        f.write(f"  Abaqus FEM time: {actual_fem_time:.1f} seconds\n")
        f.write(f"  GNN inference time: {avg_inference_time*1000:.3f} ms\n")
        f.write(f"  Speedup: {results['speedup']:.0f}× faster\n")
        f.write(f"  Time saved per prediction: {results['time_saved']:.3f} seconds\n\n")
        f.write(f"Break-even Analysis:\n")
        f.write(f"  Training time estimate: {training_time_estimate/3600:.1f} hours\n")
        f.write(f"  Break-even point: {breakeven_predictions:.0f} predictions\n")
        f.write(f"  Training ROI: Profitable after {breakeven_predictions:.0f} simulations\n")
    
    print(f"\nBenchmark results saved to: {benchmark_path}")
    print("="*60 + "\n")
    
    return results


def main():
    """Main testing function with command-line interface."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test legacy model weights with refactored code')
    
    # Model path (required)
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to legacy model checkpoint (.pt file)')
    
    # Dataset path (optional, defaults to datasets/processed_data.pt)
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to dataset (.pt file). If not provided, uses datasets/processed_data.pt')
    
    # Hyperparameters (must match training)
    parser.add_argument('--num_layers', type=int, default=5,
                       help='Number of message passing layers')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--attention_freq', type=int, default=4,
                       help='Global attention frequency')
    parser.add_argument('--dropout_rate', type=float, default=0.21887774707222715,
                       help='Dropout rate')
    
    # Dataset parameters
    parser.add_argument('--train_size', type=int, default=400,
                       help='Training set size (for splitting)')
    parser.add_argument('--test_size', type=int, default=100,
                       help='Test set size')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    
    # Other parameters
    parser.add_argument('--num_benchmark_runs', type=int, default=100,
                       help='Number of runs for inference benchmarking')
    parser.add_argument('--visualize_samples', type=int, default=25,
                       help='Number of test samples to visualize')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility (same as legacy)
    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)
    
    # Setup directories
    config_paths.setup_directories()
    checkpoint_dir = str(config_paths.BEST_MODELS_DIR)
    postprocess_dir = str(config_paths.PLOTS_DIR)
    
    # Create args object with all parameters (legacy compatibility)
    params = {
        'model_type': 'meshgraphnet',
        'num_layers': args.num_layers,
        'attention_freq': args.attention_freq,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'dropout_rate': args.dropout_rate,
        'train_size': args.train_size,
        'test_size': args.test_size,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'shuffle': True,
        'checkpoint_dir': checkpoint_dir,
        'postprocess_dir': postprocess_dir
    }
    model_args = objectview(params)
    
    print("="*70)
    print("LEGACY MODEL TESTING")
    print("="*70)
    print(f"\nModel checkpoint: {args.model_path}")
    print(f"Device: {model_args.device}")
    print(f"\nModel configuration:")
    print(f"  Layers: {args.num_layers}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Attention freq: {args.attention_freq}")
    print(f"  Dropout: {args.dropout_rate}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    if args.dataset_path:
        gnn_data_path = args.dataset_path
    else:
        gnn_data_path = str(config_paths.DATASETS_DIR / 'processed_data.pt')
    
    if not os.path.exists(gnn_data_path):
        raise FileNotFoundError(f"Dataset not found: {gnn_data_path}")
    
    print(f"Dataset path: {gnn_data_path}")
    dataset = torch.load(gnn_data_path, weights_only=False)
    
    # Shuffle dataset before splitting (CRITICAL: matches legacy behavior)
    print("Shuffling dataset with seed=5 (to match legacy training)...")
    random.shuffle(dataset)
    
    # Split dataset using provided sizes
    train_size = args.train_size
    val_size = 1
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Test set size: {test_size}")
    
    # Get statistics for normalization
    stats_list = get_stats(dataset)
    
    # Get device
    device = model_args.device
    print(f"\nUsing device: {device}")
    
    # Create model architecture
    num_node_features = test_dataset[0].x.shape[1]
    num_edge_features = test_dataset[0].edge_attr.shape[1]
    num_classes = 1
    
    print(f"\nCreating model architecture...")
    print(f"  Node features: {num_node_features}")
    print(f"  Edge features: {num_edge_features}")
    print(f"  Output dimension: {num_classes}")
    
    model = FCLGA_GraphTransformer(
        num_node_features, 
        num_edge_features, 
        args.hidden_dim, 
        num_classes, 
        model_args
    ).to(device)
    
    # Load the saved model weights
    print(f"\nLoading model weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("✓ Model weights loaded successfully!")
    
    # Benchmark inference time
    print("\n" + "="*70)
    print("STEP 1: BENCHMARKING INFERENCE TIME")
    print("="*70)
    benchmark_results = benchmark_inference_time(
        model, test_dataset, device, stats_list, 
        num_runs=args.num_benchmark_runs
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("STEP 2: EVALUATING ON TEST SET")
    print("="*70)
    final_test_loss, final_test_rmse = evaluate_final_model(
        test_dataset, model, device, stats_list, model_args
    )
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss: {final_test_loss:.5f}")
    print(f"Test RMSE: {final_test_rmse:.8f}")
    print(f"Inference Time: {benchmark_results['avg_inference_time']*1000:.3f} ms")
    print(f"Speedup vs FEM: {benchmark_results['speedup']:.0f}×")
    print(f"{'='*70}\n")
    
    # Generate visualizations on test samples
    print("\n" + "="*70)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("="*70)
    num_visualize = min(args.visualize_samples, len(test_dataset))
    print(f"Generating {num_visualize} visualizations...")
    
    for i in range(num_visualize):
        plot_name = f'test_sample_{i}_results'
        single_loader = DataLoader([test_dataset[i]], batch_size=1, shuffle=False)
        visualize(single_loader, model, postprocess_dir, plot_name, stats_list)
        
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{num_visualize} visualizations")
    
    print(f"\n✓ All visualizations saved to: {postprocess_dir}")
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
