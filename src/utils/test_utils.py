"""
Testing Utilities for FCLGA GraphTransformer

This module provides utility functions for model evaluation including:
- Inference benchmarking
- Model evaluation with RMSE metrics
- Visualization helpers

Author: Luca Patrignani
Institution: Imperial College London
"""

import torch
import numpy as np
import os
import time

from src.utils.data_utils import unnormalize
from src.utils.visualization import plot_results
from config import paths as config_paths


class ObjectView:
    """Convert dictionary to object with attribute access."""

    def __init__(self, dictionary):
        """
        Initialize from dictionary.

        Args:
            dictionary (dict): Dictionary to convert to object attributes.
        """
        self.__dict__ = dictionary


def visualize_sample(model, dataset, sample_index, file_dir, plot_name, stats_list):
    """
    Generate visualization for a single sample.

    Args:
        model: Trained FCLGA_GraphTransformer model.
        dataset: Dataset containing samples to visualize.
        sample_index (int): Index of sample to visualize.
        file_dir (str): Directory to save visualization files.
        plot_name (str): Base name for saved plots.
        stats_list (list): Normalization statistics [mean_x, std_x, mean_edge,
                          std_edge, mean_y, std_y].
    """
    model.eval()
    device = next(model.parameters()).device

    # Unpack statistics
    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y = stats_list

    # Move statistics to device
    mean_vec_x = mean_vec_x.to(device)
    std_vec_x = std_vec_x.to(device)
    mean_vec_edge = mean_vec_edge.to(device)
    std_vec_edge = std_vec_edge.to(device)
    mean_vec_y = mean_vec_y.to(device)
    std_vec_y = std_vec_y.to(device)

    # Get sample and move to device
    sample = dataset[sample_index].to(device)

    # Generate prediction
    with torch.no_grad():
        pred = model(sample, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
        pred = unnormalize(pred, mean_vec_y, std_vec_y)

    # Generate visualization
    plot_results(sample, pred, file_dir, plot_name)


def evaluate_model(dataset, model, device, stats_list, postprocess_dir):
    """
    Evaluate model on dataset with detailed RMSE metrics.

    Calculates per-sample and global RMSE with masking for padded nodes.
    Saves detailed results to text file.

    Args:
        dataset: PyTorch Geometric dataset to evaluate.
        model: Trained FCLGA_GraphTransformer model.
        device (str): Device to run evaluation on ('cuda' or 'cpu').
        stats_list (list): Normalization statistics.
        postprocess_dir (str): Directory to save evaluation results.

    Returns:
        tuple: (mean_loss, global_rmse) - Average loss and global RMSE across all samples.
    """
    # Unpack statistics
    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y = stats_list

    # Move statistics to device
    mean_vec_x = mean_vec_x.to(device)
    std_vec_x = std_vec_x.to(device)
    mean_vec_edge = mean_vec_edge.to(device)
    std_vec_edge = std_vec_edge.to(device)
    mean_vec_y = mean_vec_y.to(device)
    std_vec_y = std_vec_y.to(device)

    model.eval()

    total_loss = 0
    num_samples = 0
    individual_rmses = []

    # Store all predictions and ground truths for global RMSE
    all_preds = []
    all_gts = []

    print("\nEvaluating on test set...")
    print("=" * 70)

    with torch.no_grad():
        for data in dataset:
            data = data.to(device)

            # Forward pass
            out = model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)

            # Calculate loss
            loss = model.loss(out, data, mean_vec_y, std_vec_y)
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
                print(f"Sample {num_samples:3d}: RMSE = {sample_rmse:.8f}")
            else:
                print(f"Sample {num_samples:3d}: No non-zero nodes found")
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
    print(f"\n{'=' * 70}")
    print("TEST SET EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Number of test samples: {num_samples}")
    print(f"Mean Test Loss: {mean_loss:.5f}")
    print("\n--- RMSE METRICS ---")
    print(f"Global RMSE (all nodes, all samples): {global_rmse:.8f}")
    print(f"Mean of per-sample RMSEs: {mean_sample_rmse:.8f}")
    print(f"RMSE Std Dev: {rmse_std:.8f}")
    print(f"RMSE Range: [{rmse_min:.8f}, {rmse_max:.8f}]")
    print(f"{'=' * 70}\n")

    # Save detailed results
    results_path = os.path.join(postprocess_dir, 'test_results_detailed.txt')
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of test samples: {num_samples}\n")
        f.write(f"Mean Test Loss: {mean_loss:.5f}\n\n")
        f.write("--- RMSE METRICS ---\n")
        f.write(f"Global RMSE (all nodes, all samples): {global_rmse:.8f}\n")
        f.write(f"Mean of per-sample RMSEs: {mean_sample_rmse:.8f}\n")
        f.write(f"RMSE Std Dev: {rmse_std:.8f}\n")
        f.write(f"RMSE Range: [{rmse_min:.8f}, {rmse_max:.8f}]\n\n")
        f.write("Individual Sample RMSEs:\n")
        for i, rmse in enumerate(individual_rmses):
            f.write(f"  Sample {i}: {rmse:.8f}\n")

    print(f"Detailed results saved to: {results_path}")

    return mean_loss, global_rmse


def benchmark_inference(model, dataset, device, stats_list, postprocess_dir, num_runs=100):
    """
    Benchmark model inference time and compare with FEM.

    Performs warmup runs followed by timed inference iterations to measure
    average prediction time. Compares with typical FEM simulation time.

    Args:
        model: Trained FCLGA_GraphTransformer model.
        dataset: Dataset to benchmark on (uses first sample).
        device (str): Device to run on ('cuda' or 'cpu').
        stats_list (list): Normalization statistics.
        postprocess_dir (str): Directory to save benchmark results.
        num_runs (int, optional): Number of inference iterations. Defaults to 100.

    Returns:
        dict: Dictionary containing benchmark results including average time,
              speedup vs FEM, and break-even analysis.
    """
    # Unpack statistics
    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y = stats_list

    # Move statistics to device
    mean_vec_x = mean_vec_x.to(device)
    std_vec_x = std_vec_x.to(device)
    mean_vec_edge = mean_vec_edge.to(device)
    std_vec_edge = std_vec_edge.to(device)
    mean_vec_y = mean_vec_y.to(device)
    std_vec_y = std_vec_y.to(device)

    model.eval()

    print("\n" + "=" * 60)
    print("INFERENCE TIME BENCHMARK")
    print("=" * 60 + "\n")

    # Use first sample for benchmarking
    test_sample = dataset[0].to(device)

    # Warmup runs (ensure GPU is ready)
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

    # FEM baseline (typical Abaqus simulation time)
    actual_fem_time = 120.0  # seconds

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

    # Print results
    print("\nGNN Inference Results:")
    print(
        f"  Average inference time: {avg_inference_time * 1000:.3f} ms "
        f"({avg_inference_time:.6f} s)"
    )
    print(f"  Number of runs: {num_runs}")
    print(f"  Total time: {total_inference_time:.3f} seconds")
    print(
        f"  Inference frequency: "
        f"{1 / avg_inference_time:.1f} predictions/second"
    )

    # Save results to file
    benchmark_path = os.path.join(postprocess_dir, 'inference_benchmark.txt')
    with open(benchmark_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("INFERENCE TIME BENCHMARK\n")
        f.write("=" * 60 + "\n\n")
        f.write("GNN Inference Results:\n")
        f.write(
            f"  Average inference time: {avg_inference_time * 1000:.3f} ms "
            f"({avg_inference_time:.6f} s)\n"
        )
        f.write(f"  Number of runs: {num_runs}\n")
        f.write(f"  Total time: {total_inference_time:.3f} seconds\n")
        inf_freq = 1 / avg_inference_time
        f.write(
            f"  Inference frequency: {inf_freq:.1f} predictions/second\n"
        )

    print(f"\nBenchmark results saved to: {benchmark_path}")
    print("=" * 60 + "\n")

    return results
