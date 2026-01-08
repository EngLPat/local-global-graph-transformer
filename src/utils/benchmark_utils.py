"""Benchmarking utilities for FCLGA GraphTransformer.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import copy
import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.utils.optimizer_utils import build_optimizer

# Note: create_journal_quality_timing_plots may need to be defined or imported correctly
# from src.utils.visualization import create_journal_quality_timing_plots


def create_summary_table(df, results_folder):
    """
    Create a formatted summary table for the paper.
    """
    # Create a comprehensive summary
    summary_data = []

    baseline_time = df[df["expected_calls"] == 0]["time_per_epoch"].iloc[0]

    for _, row in df.iterrows():
        summary_data.append(
            {
                "Configuration": row["description"],
                "Attention Calls": int(row["expected_calls"]),
                "Time per Epoch (s)": f"{row['time_per_epoch']:.3f}",
                "Overhead": f"{row['overhead_vs_baseline']:.2f}×",
                "Additional Time (s)": f"{row['time_per_epoch'] - baseline_time:.3f}",
                "Efficiency (s/call)": f"{(row['time_per_epoch'] - baseline_time) / max(row['expected_calls'], 1):.3f}"
                if row["expected_calls"] > 0
                else "N/A",
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    csv_path = os.path.join(results_folder, "timing_analysis_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    # Create LaTeX table
    latex_table = summary_df.to_latex(
        index=False,
        escape=False,
        caption="Computational timing analysis for different global attention frequencies.",
        label="tab:timing_analysis",
    )

    # Save LaTeX table
    latex_path = os.path.join(results_folder, "timing_analysis_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)

    print("\nSummary table saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  LaTeX: {latex_path}")

    return summary_df


def save_timing_results(df, results_folder):
    """
    Save timing results with journal-quality visualizations.
    """
    # Save raw data
    csv_path = os.path.join(results_folder, "timing_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Create journal-quality plots
    fig = create_journal_quality_timing_plots(df, results_folder)

    # Create summary table
    summary_df = create_summary_table(df, results_folder)

    # Print results summary
    print("\n" + "=" * 80)
    print("JOURNAL-QUALITY TIMING ANALYSIS COMPLETE")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    return fig, summary_df


def benchmark_attention_frequencies(dataset, device, stats_list, base_args, num_epochs=5):
    """
    Benchmark different attention frequencies to measure timing and complexity.
    """
    # Test configurations: attention_freq values and their descriptions
    test_configs = [
        (999, "No attention", 0),  # No attention (freq > num_layers)
        (6, "Every 6 layers", 1),  # Every 6 layers = 1 call
        (3, "Every 3 layers", 2),  # Every 3 layers = 2 calls
        (2, "Every 2 layers", 3),  # Every 2 layers = 3 calls
        (1, "Every 1 layer", 6),  # Every layer = 6 calls
    ]

    results = []

    # Use smaller dataset for timing tests
    test_dataset = dataset[:400]  # Use first 50 samples for quick testing

    for freq, description, expected_calls in test_configs:
        print(f"\nTesting: {description} (freq={freq})")

        # Create args for this configuration
        test_args = copy.deepcopy(base_args)
        test_args.attention_freq = freq
        test_args.epochs = num_epochs

        # Create model
        num_node_features = test_dataset[0].x.shape[1]
        num_edge_features = test_dataset[0].edge_attr.shape[1]
        model = MeshGraphNet(
            num_node_features, num_edge_features, test_args.hidden_dim, 1, test_args
        ).to(device)

        # Create data loader
        loader = DataLoader(test_dataset, batch_size=test_args.batch_size, shuffle=False)

        # Get stats
        [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
        stats_gpu = [s.to(device) for s in stats_list]

        # Build optimizer
        scheduler, opt = build_optimizer(test_args, model.parameters())

        # Warm up GPU
        model.train()
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, *stats_gpu[:4])
            loss = model.loss(pred, batch, *stats_gpu[4:])
            break

        # Time the training
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()

        total_loss = 0
        num_batches = 0

        for epoch in range(num_epochs):
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                pred = model(batch, *stats_gpu[:4])
                loss = model.loss(pred, batch, *stats_gpu[4:])
                loss.backward()
                opt.step()
                total_loss += loss.item()
                num_batches += 1

        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        time_per_epoch = total_time / num_epochs
        avg_loss = total_loss / num_batches

        # Store results
        result = {
            "attention_freq": freq,
            "description": description,
            "expected_calls": expected_calls,
            "time_per_epoch": time_per_epoch,
            "total_time": total_time,
            "avg_loss": avg_loss,
            "num_epochs": num_epochs,
        }
        results.append(result)

        print(f"  Time per epoch: {time_per_epoch:.3f}s")
        print(f"  Average loss: {avg_loss:.6f}")

        # Clean up
        del model, loader
        torch.cuda.empty_cache() if device == "cuda" else None

    return results


def analyze_timing_results(results):
    """
    Analyze and display timing results with complexity calculations.
    """
    df = pd.DataFrame(results)

    # Calculate overhead vs baseline (no attention)
    baseline_time = df[df["expected_calls"] == 0]["time_per_epoch"].iloc[0]
    df["overhead_vs_baseline"] = df["time_per_epoch"] / baseline_time

    print("\n" + "=" * 80)
    print("TIMING BENCHMARK RESULTS")
    print("=" * 80)

    print(f"{'Description':<20} {'Calls':<6} {'Time/Epoch':<12} {'Overhead':<10}")
    print("-" * 50)

    for _, row in df.iterrows():
        print(
            f"{row['description']:<20} {row['expected_calls']:<6} "
            f"{row['time_per_epoch']:.3f}s{'':<6} {row['overhead_vs_baseline']:.2f}×"
        )

    # Calculate theoretical operations (assuming 1000 nodes, 3000 edges, hidden_dim=48)
    hidden_dim = 48  # From your config
    num_nodes = 1000  # Approximate
    num_edges = 3000  # Approximate
    num_layers = 6

    print(
        f"\n{'Description':<20} {'Message Passing Ops':<20} {'Attention Ops':<15} {'Total Ops':<15}"
    )
    print("-" * 70)

    message_passing_ops = num_layers * num_edges * (hidden_dim**2)

    for _, row in df.iterrows():
        attention_ops = row["expected_calls"] * (num_nodes**2) * hidden_dim
        total_ops = message_passing_ops + attention_ops

        print(
            f"{row['description']:<20} {message_passing_ops / 1e6:.1f}M{'':<15} "
            f"{attention_ops / 1e6:.1f}M{'':<10} {total_ops / 1e6:.1f}M"
        )

    return df


def benchmark_inference_time(model, test_dataset, device, stats_list, num_runs=100):
    """
    Benchmark GNN inference time for fair comparison with FEM simulation time.

    Args:
        model: Trained GNN model
        test_dataset: Test dataset
        device: Computing device
        stats_list: Normalization statistics
        num_runs: Number of inference runs for averaging

    Returns:
        dict: Timing results and speedup analysis
    """
    print(f"\n{'=' * 60}")
    print("INFERENCE TIME BENCHMARK")
    print(f"{'=' * 60}")

    # Get statistics
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    stats_gpu = [s.to(device) for s in stats_list]

    # Prepare single sample for inference
    sample = test_dataset[0].to(device)
    model.eval()

    # Warm up GPU
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample, *stats_gpu)

    # Time GNN inference
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            prediction = model(sample, *stats_gpu)

    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()

    # Calculate average inference time
    total_inference_time = end_time - start_time
    avg_inference_time = total_inference_time / num_runs

    # Estimate FEM simulation time (you should replace this with actual FEM timing)
    # These are typical values - you should measure actual FEM times for your meshes
    estimated_fem_times = {
        "simple_mesh": 300,  # 5 minutes for simple mesh
        "medium_mesh": 1800,  # 30 minutes for medium complexity
        "complex_mesh": 7200,  # 2 hours for complex mesh
        "your_mesh": 600,  # Estimate for your specific mesh - UPDATE THIS
    }

    results = {
        "gnn_inference_time": avg_inference_time,
        "num_runs": num_runs,
        "total_time": total_inference_time,
        "speedup_analysis": {},
    }

    print("GNN Inference Results:")
    print(f"  Average inference time: {avg_inference_time:.6f} seconds")
    print(f"  Total time for {num_runs} runs: {total_inference_time:.3f} seconds")
    print(f"  Inference frequency: {1 / avg_inference_time:.1f} predictions/second")

    print("\nSpeedup Analysis vs FEM:")
    print(f"{'FEM Type':<15} {'FEM Time (s)':<15} {'Speedup':<15} {'Time Saved':<15}")
    print("-" * 65)

    for fem_type, fem_time in estimated_fem_times.items():
        speedup = fem_time / avg_inference_time
        time_saved = fem_time - avg_inference_time
        results["speedup_analysis"][fem_type] = {
            "fem_time": fem_time,
            "speedup": speedup,
            "time_saved": time_saved,
        }
        print(f"{fem_type:<15} {fem_time:<15.1f} {speedup:<15.0f}× {time_saved:<15.1f}s")

    # Break-even analysis for training cost
    training_time_estimate = 3600  # 1 hour estimate - update with your actual training time
    print("\nBreak-even Analysis:")
    print(f"Training time estimate: {training_time_estimate:.0f} seconds")

    for fem_type, analysis in results["speedup_analysis"].items():
        time_saved_per_pred = analysis["time_saved"]
        break_even_predictions = (
            training_time_estimate / time_saved_per_pred
            if time_saved_per_pred > 0
            else float("inf")
        )
        print(f"  {fem_type}: Break-even after {break_even_predictions:.0f} predictions")

    return results


def save_speedup_analysis(results, results_folder):
    """Save speedup analysis results to files."""

    # Save as JSON
    json_path = os.path.join(results_folder, "speedup_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Speedup comparison
    fem_types = list(results["speedup_analysis"].keys())
    speedups = [results["speedup_analysis"][ft]["speedup"] for ft in fem_types]

    bars = ax1.bar(fem_types, speedups, color="skyblue", alpha=0.7, edgecolor="black")
    ax1.set_ylabel("Speedup Factor (×)")
    ax1.set_title("GNN vs FEM Speedup")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{speedup:.0f}×",
            ha="center",
            va="bottom",
        )

    # Plot 2: Time comparison
    fem_times = [results["speedup_analysis"][ft]["fem_time"] for ft in fem_types]
    gnn_time = results["gnn_inference_time"]

    x = range(len(fem_types))
    ax2.bar([i - 0.2 for i in x], fem_times, 0.4, label="FEM", color="red", alpha=0.7)
    ax2.bar(
        [i + 0.2 for i in x], [gnn_time] * len(fem_types), 0.4, label="GNN", color="blue", alpha=0.7
    )

    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Absolute Time Comparison")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(fem_types)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(results_folder, "speedup_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSpeedup analysis saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Plot: {plot_path}")
