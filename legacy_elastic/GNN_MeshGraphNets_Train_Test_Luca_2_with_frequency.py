"""
*******************************************************************************
*                                                                             *
*   AUTHOR: Luca Patrignani - PhD candidate Imperial College London           *
*   TITLE: GNN MeshGraphNets Training and Testing                             *
*   DATE: 24/02/2025                                                          *
*                                                                             *
*******************************************************************************
*                                                                             *
*  Description:                                                               *
*  ============                                                               *
*  This Python script was meticulously crafted to design and develop a GNN    *
*  to solve a mesh graph problem using PyTorch and PyTorch Geometric.         *
*  This version, GNN_MeshGraphNets_Train_Test_Luca, is a variation of the     *
*  original GNN_MeshGraphNets_Train_Test script, utilizing Luca's model.      *
*                                                                             *
*  Rights:                                                                    *
*  ======                                                                     *
*  All rights to this code are reserved.                                      *
*                                                                             *
*******************************************************************************
"""


# to check if I remove the global attention, then how fast is it?

import torch
import random
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU, Dropout
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import tri as mtri
from torch.nn import PReLU

import numpy as np
import time
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy
import matplotlib.pyplot as plt
import os
import h5py
# import tensorflow.compat.v1 as tf
import functools
import json
from torch_geometric.data import Data
import enum
import math

import os
import datetime
from pathlib import Path

import time
import pandas as pd
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import numpy as np

def create_journal_quality_timing_plots(df, results_folder):
    """
    Create high-quality timing analysis plots suitable for journal publication.
    """
    # Set style for publication quality
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], 
                         hspace=0.3, wspace=0.25)
    
    # Define colors for consistency
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
    
    # Plot 1: Training Time vs Attention Frequency (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    x_vals = df['expected_calls'].values
    y_vals = df['time_per_epoch'].values
    
    # Plot line with markers
    ax1.plot(x_vals, y_vals, 'o-', linewidth=3, markersize=10, 
             color=colors[0], markerfacecolor='white', markeredgewidth=2, 
             markeredgecolor=colors[0])
    
    # Add value labels
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        ax1.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontsize=11, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                             edgecolor=colors[0], alpha=0.8))
    
    ax1.set_xlabel('Number of Global Attention Calls', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time per Epoch (seconds)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Training Time vs Attention Frequency', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-0.3, 6.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 2: Computational Overhead (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    bars = ax2.bar(range(len(df)), df['overhead_vs_baseline'], 
                   color=[colors[i] for i in range(len(df))], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, overhead) in enumerate(zip(bars, df['overhead_vs_baseline'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05, 
                f'{overhead:.2f}×', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Attention Configuration', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Overhead vs Baseline (×)', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Computational Overhead Analysis', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([f"{int(calls)} calls" for calls in df['expected_calls']], 
                       rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 3: Linear Scaling Analysis (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Theoretical linear fit
    x_theory = np.array([0, 1, 2, 3, 6])
    baseline = df[df['expected_calls'] == 0]['time_per_epoch'].iloc[0]
    
    # Calculate linear regression
    slope = (df['time_per_epoch'].iloc[-1] - baseline) / 6  # slope per attention call
    y_theory = baseline + slope * x_theory
    
    # Plot empirical data
    ax3.plot(x_vals, y_vals, 'o', markersize=12, color=colors[1], 
             label='Empirical Data', markerfacecolor='white', 
             markeredgewidth=3, markeredgecolor=colors[1])
    
    # Plot theoretical line
    ax3.plot(x_theory, y_theory, '--', linewidth=3, color=colors[2], 
             label='Linear Fit', alpha=0.8)
    
    # Calculate R-squared
    y_pred = baseline + slope * x_vals
    ss_res = np.sum((y_vals - y_pred) ** 2)
    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    ax3.text(0.05, 0.95, f'R² = {r_squared:.3f}\nSlope = {slope:.3f} s/call', 
             transform=ax3.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                      edgecolor='gray', alpha=0.9))
    
    ax3.set_xlabel('Number of Global Attention Calls', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Time per Epoch (seconds)', fontsize=14, fontweight='bold')
    ax3.set_title('(c) Linear Scaling Validation', fontsize=16, fontweight='bold', pad=20)
    ax3.legend(fontsize=12, loc='lower right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 4: Operation Count Analysis (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate theoretical operations
    hidden_dim = 48
    num_nodes = 1000
    num_edges = 3000
    num_layers = 6
    
    message_passing_ops = num_layers * num_edges * (hidden_dim ** 2) / 1e6  # Convert to millions
    attention_ops = df['expected_calls'] * (num_nodes ** 2) * hidden_dim / 1e6
    
    width = 0.6
    x_pos = np.arange(len(df))
    
    bars1 = ax4.bar(x_pos, [message_passing_ops] * len(df), width, 
                   label='Message Passing', color=colors[0], alpha=0.8)
    bars2 = ax4.bar(x_pos, attention_ops, width, bottom=[message_passing_ops] * len(df),
                   label='Global Attention', color=colors[3], alpha=0.8)
    
    # Add total operation labels
    for i, (mp, att) in enumerate(zip([message_passing_ops] * len(df), attention_ops)):
        total = mp + att
        ax4.text(i, total + 5, f'{total:.0f}M', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Attention Configuration', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Operations (Millions)', fontsize=14, fontweight='bold')
    ax4.set_title('(d) Theoretical Operation Count', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{int(calls)} calls" for calls in df['expected_calls']], 
                       rotation=45, ha='right')
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 5: Efficiency Analysis (Bottom Spanning)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Calculate efficiency metrics
    operations_total = message_passing_ops + attention_ops
    efficiency = y_vals / operations_total * 1000  # Time per million operations
    
    # Create dual y-axis plot
    ax5_twin = ax5.twinx()
    
    # Plot bars for total operations
    bars = ax5.bar(x_pos, operations_total, width=0.6, alpha=0.6, 
                  color=colors[4], label='Total Operations')
    
    # Plot line for timing efficiency
    line = ax5_twin.plot(x_pos, efficiency, 'o-', linewidth=3, markersize=8, 
                        color=colors[1], label='Time per MOp')
    
    ax5.set_xlabel('Number of Global Attention Calls', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Total Operations (Millions)', fontsize=14, fontweight='bold', color=colors[4])
    ax5_twin.set_ylabel('Time per Million Operations (ms)', fontsize=14, fontweight='bold', color=colors[1])
    ax5.set_title('(e) Computational Efficiency Analysis', fontsize=16, fontweight='bold', pad=20)
    
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f"{int(calls)}" for calls in df['expected_calls']])
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax5.tick_params(axis='both', which='major', labelsize=12)
    ax5_twin.tick_params(axis='both', which='major', labelsize=12)
    
    # Color the y-axis labels to match the data
    ax5.tick_params(axis='y', colors=colors[4])
    ax5_twin.tick_params(axis='y', colors=colors[1])
    
    # Add combined legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    
    # Overall title
    fig.suptitle('Computational Complexity Analysis: Global Attention Impact on MeshGraphNet Training', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save high-resolution figure
    plot_path = os.path.join(results_folder, 'journal_timing_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Also save as PDF for vector graphics
    pdf_path = os.path.join(results_folder, 'journal_timing_analysis.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    print(f"High-quality plots saved to:")
    print(f"  PNG: {plot_path}")
    print(f"  PDF: {pdf_path}")
    
    return fig

def create_summary_table(df, results_folder):
    """
    Create a formatted summary table for the paper.
    """
    # Create a comprehensive summary
    summary_data = []
    
    baseline_time = df[df['expected_calls'] == 0]['time_per_epoch'].iloc[0]
    
    for _, row in df.iterrows():
        summary_data.append({
            'Configuration': row['description'],
            'Attention Calls': int(row['expected_calls']),
            'Time per Epoch (s)': f"{row['time_per_epoch']:.3f}",
            'Overhead': f"{row['overhead_vs_baseline']:.2f}×",
            'Additional Time (s)': f"{row['time_per_epoch'] - baseline_time:.3f}",
            'Efficiency (s/call)': f"{(row['time_per_epoch'] - baseline_time) / max(row['expected_calls'], 1):.3f}" if row['expected_calls'] > 0 else "N/A"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(results_folder, 'timing_analysis_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    
    # Create LaTeX table
    latex_table = summary_df.to_latex(index=False, escape=False, 
                                     caption="Computational timing analysis for different global attention frequencies.",
                                     label="tab:timing_analysis")
    
    # Save LaTeX table
    latex_path = os.path.join(results_folder, 'timing_analysis_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    
    print(f"\nSummary table saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  LaTeX: {latex_path}")
    
    return summary_df

# Update your save_timing_results function:
def save_timing_results(df, results_folder):
    """
    Save timing results with journal-quality visualizations.
    """
    # Save raw data
    csv_path = os.path.join(results_folder, 'timing_benchmark.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create journal-quality plots
    fig = create_journal_quality_timing_plots(df, results_folder)
    
    # Create summary table
    summary_df = create_summary_table(df, results_folder)
    
    # Print results summary
    print("\n" + "="*80)
    print("JOURNAL-QUALITY TIMING ANALYSIS COMPLETE")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    return fig, summary_df

def benchmark_attention_frequencies(dataset, device, stats_list, base_args, num_epochs=5):
    """
    Benchmark different attention frequencies to measure timing and complexity.
    """
    # Test configurations: attention_freq values and their descriptions
    test_configs = [
        (999, "No attention", 0),      # No attention (freq > num_layers)
        (6, "Every 6 layers", 1),      # Every 6 layers = 1 call
        (3, "Every 3 layers", 2),      # Every 3 layers = 2 calls  
        (2, "Every 2 layers", 3),      # Every 2 layers = 3 calls
        (1, "Every 1 layer", 6),       # Every layer = 6 calls
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
        model = MeshGraphNet(num_node_features, num_edge_features, 
                           test_args.hidden_dim, 1, test_args).to(device)
        
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
        torch.cuda.synchronize() if device == 'cuda' else None
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
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        time_per_epoch = total_time / num_epochs
        avg_loss = total_loss / num_batches
        
        # Store results
        result = {
            'attention_freq': freq,
            'description': description,
            'expected_calls': expected_calls,
            'time_per_epoch': time_per_epoch,
            'total_time': total_time,
            'avg_loss': avg_loss,
            'num_epochs': num_epochs
        }
        results.append(result)
        
        print(f"  Time per epoch: {time_per_epoch:.3f}s")
        print(f"  Average loss: {avg_loss:.6f}")
        
        # Clean up
        del model, loader
        torch.cuda.empty_cache() if device == 'cuda' else None
    
    return results

def analyze_timing_results(results):
    """
    Analyze and display timing results with complexity calculations.
    """
    df = pd.DataFrame(results)
    
    # Calculate overhead vs baseline (no attention)
    baseline_time = df[df['expected_calls'] == 0]['time_per_epoch'].iloc[0]
    df['overhead_vs_baseline'] = df['time_per_epoch'] / baseline_time
    
    print("\n" + "="*80)
    print("TIMING BENCHMARK RESULTS")
    print("="*80)
    
    print(f"{'Description':<20} {'Calls':<6} {'Time/Epoch':<12} {'Overhead':<10}")
    print("-" * 50)
    
    for _, row in df.iterrows():
        print(f"{row['description']:<20} {row['expected_calls']:<6} "
              f"{row['time_per_epoch']:.3f}s{'':<6} {row['overhead_vs_baseline']:.2f}×")
    
    # Calculate theoretical operations (assuming 1000 nodes, 3000 edges, hidden_dim=48)
    hidden_dim = 48  # From your config
    num_nodes = 1000  # Approximate
    num_edges = 3000  # Approximate
    num_layers = 6
    
    print(f"\n{'Description':<20} {'Message Passing Ops':<20} {'Attention Ops':<15} {'Total Ops':<15}")
    print("-" * 70)
    
    message_passing_ops = num_layers * num_edges * (hidden_dim ** 2)
    
    for _, row in df.iterrows():
        attention_ops = row['expected_calls'] * (num_nodes ** 2) * hidden_dim
        total_ops = message_passing_ops + attention_ops
        
        print(f"{row['description']:<20} {message_passing_ops/1e6:.1f}M{'':<15} "
              f"{attention_ops/1e6:.1f}M{'':<10} {total_ops/1e6:.1f}M")
    
    return df

# # def save_timing_results(df, results_folder):
#     """
#     Save timing results to CSV and create plots.
#     """
#     # Save to CSV
#     csv_path = os.path.join(results_folder, 'timing_benchmark.csv')
#     df.to_csv(csv_path, index=False)
#     print(f"\nResults saved to: {csv_path}")
    
#     # Create plots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
#     # Plot 1: Time per epoch vs number of attention calls
#     ax1.plot(df['expected_calls'], df['time_per_epoch'], 'bo-', linewidth=2, markersize=8)
#     ax1.set_xlabel('Number of Global Attention Calls')
#     ax1.set_ylabel('Time per Epoch (seconds)')
#     ax1.set_title('Training Time vs Attention Frequency')
#     ax1.grid(True, alpha=0.3)
    
#     # Add annotations
#     for _, row in df.iterrows():
#         ax1.annotate(f"{row['time_per_epoch']:.2f}s", 
#                     (row['expected_calls'], row['time_per_epoch']),
#                     textcoords="offset points", xytext=(0,10), ha='center')
    
#     # Plot 2: Overhead vs baseline
#     ax2.bar(range(len(df)), df['overhead_vs_baseline'], color='skyblue', alpha=0.7)
#     ax2.set_xlabel('Configuration')
#     ax2.set_ylabel('Overhead vs Baseline (×)')
#     ax2.set_title('Computational Overhead')
#     ax2.set_xticks(range(len(df)))
#     ax2.set_xticklabels([f"{row['expected_calls']} calls" for _, row in df.iterrows()], 
#                        rotation=45)
#     ax2.grid(True, alpha=0.3, axis='y')
    
#     # Add value labels on bars
#     for i, (_, row) in enumerate(df.iterrows()):
#         ax2.text(i, row['overhead_vs_baseline'] + 0.05, f"{row['overhead_vs_baseline']:.2f}×",
#                 ha='center', va='bottom')
    
#     plt.tight_layout()
#     plot_path = os.path.join(results_folder, 'timing_benchmark.png')
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"Plot saved to: {plot_path}")

def create_results_folder():
    # Get current timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the folder name with the timestamp
    result_dir = f"0_standard_{timestamp}"
    
    # Create the full path (assuming you want it in the current directory)
    full_path = Path(os.getcwd()) / "results" / result_dir
    
    # Create the directory if it doesn't exist
    os.makedirs(full_path, exist_ok=True)
    
    return full_path

# Use it like this:
results_folder = create_results_folder()

# Define directories for datasets, checkpoints, and postprocessing
root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir, 'datasets')

# Fix: Use results_folder for output directories instead of root_dir
checkpoint_dir = os.path.join(results_folder, 'best_models')
postprocess_dir = os.path.join(results_folder, 'plots')

# Create these subdirectories
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(postprocess_dir, exist_ok=True)


gnn_data_path = os.path.join(dataset_dir, 'processed_data.pt')
data = torch.load(gnn_data_path,weights_only=False)

#Define the list that will return the data graphs
data_list = []


    
def visualize(loader_original, model, file_dir, plot_name, stats_list, sample_index=0):
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

def plot_results(data, prediction, path, name, remote_stress=0.01):
    print('Generating stress fields...')
    fig, axes = plt.subplots(1, 4, figsize=(26, 5))  # Adjust subplot for an additional plot
    
    # Ensure data is on CPU for matplotlib processing
    pos = data.mesh_pos.cpu().numpy()
    faces = data.cells.cpu().numpy()
    
    # Prepare the ground truth, prediction, error, and relative error data
    gs_stress = data.y[:, 0].cpu().numpy()
    pred_stress = prediction[:, 0].cpu().numpy()
    error_stress = pred_stress - gs_stress
    epsilon = 1e-10
    # relative_error_stress = ((pred_stress - gs_stress) / (np.abs(gs_stress) + epsilon)) * 100
    relative_error_stress = ((pred_stress - gs_stress) / remote_stress) * 100  # Relative error calculation

    # Print diagnostic information
    print(f"Ground truth range: {gs_stress.min():.6f} to {gs_stress.max():.6f}")
    print(f"Prediction range: {pred_stress.min():.6f} to {pred_stress.max():.6f}")
    print(f"Absolute error range: {error_stress.min():.6f} to {error_stress.max():.6f}")
    print(f"Relative error range: {relative_error_stress.min():.2f}% to {relative_error_stress.max():.2f}%")

    # Find common min and max for consistent coloring across first 3 plots
    vmin = min(gs_stress.min(), pred_stress.min(), error_stress.min())
    vmax = max(gs_stress.max(), pred_stress.max(), error_stress.max())
    
    titles = ['Ground Truth', 'Prediction', 'Error', 'Relative Error (%)']
    stresses = [gs_stress, pred_stress, error_stress, relative_error_stress]
    max_stress = max(gs_stress.max(), pred_stress.max())
    max_error = abs(error_stress).max()
    clim = [(0, max_stress), (0, max_stress), (-0.5*max_stress, 0.5*max_stress), (-20, 20)]

    for ax, stress, title, clims in zip(axes, stresses, titles, clim):
        ax.cla()
        ax.set_aspect('equal')
        ax.set_axis_off()
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        cmap_choice = 'viridis' if title != 'Relative Error (%)' else 'coolwarm'
        mesh_plot = ax.tripcolor(triang, stress, shading='flat', cmap=cmap_choice, vmin=clims[0], vmax=clims[1])
        ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
        if title == 'Relative Error (%)':
            clb.set_label('% Error')  # Label the colorbar meaningfully for relative error

    # Save the plotted data
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save numerical data for later use
    data_path = os.path.join(path, name + '_data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # Save arrays for later analysis in other formats
    np.save(os.path.join(data_path, 'mesh_positions.npy'), pos)
    np.save(os.path.join(data_path, 'mesh_faces.npy'), faces)
    np.save(os.path.join(data_path, 'ground_truth.npy'), gs_stress)
    np.save(os.path.join(data_path, 'prediction.npy'), pred_stress)
    np.save(os.path.join(data_path, 'error.npy'), error_stress)
    np.save(os.path.join(data_path, 'relative_error.npy'), relative_error_stress)
    
    # Save as CSV format for easy import to other software
    np.savetxt(os.path.join(data_path, 'results.csv'), 
               np.column_stack((pos, gs_stress, pred_stress, error_stress, relative_error_stress)),
               delimiter=',', 
               header='x,y,ground_truth,prediction,error,relative_error')
    
    plot_path = os.path.join(path, name + '_comparison.png')
    plt.savefig(plot_path)
    plt.show()


file_path = os.path.join(dataset_dir, 'processed_data.pt')
dataset_full_timesteps = torch.load(gnn_data_path, weights_only=False)
dataset = torch.load(file_path, weights_only=False)
if not isinstance(dataset, list):
    dataset = [dataset]
dataset = dataset[:1]

print(dataset)
len(dataset_full_timesteps)/5

def normalize(to_normalize, mean_vec, std_vec):
    # print(f"Shape of to_normalize before normalization: {to_normalize.shape}")
    # print(f"Shape of mean_vec: {mean_vec.shape}")
    # print(f"Shape of std_vec: {std_vec.shape}")
    normalized = (to_normalize - mean_vec) / std_vec
    # print(f"Shape of normalized: {normalized.shape}")
    return normalized

def unnormalize(to_unnormalize,mean_vec,std_vec):
    return to_unnormalize*std_vec+mean_vec

def get_stats(data_list):
    '''
    Method for normalizing processed datasets. Given  the processed data_list,
    calculates the mean and standard deviation for the node features, edge features,
    and node outputs, and normalizes these using the calculated statistics.
    '''

    #mean and std of the node features are calculated
    mean_vec_x=torch.zeros(data_list[0].x.shape[1:])
    std_vec_x=torch.zeros(data_list[0].x.shape[1:])

    #mean and std of the edge features are calculated
    mean_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])

    #mean and std of the output parameters are calculated
    mean_vec_y=torch.zeros(data_list[0].y.shape[1:])
    std_vec_y=torch.zeros(data_list[0].y.shape[1:])

    #Define the maximum number of accumulations to perform such that we do
    #not encounter memory issues
    max_accumulations = 10**6

    #Define a very small value for normalizing to
    eps=torch.tensor(1e-8)

    #Define counters used in normalization
    num_accs_x = 0
    num_accs_edge=0
    num_accs_y=0

    #Iterate through the data in the list to accumulate statistics
    for dp in data_list:

        #Add to the
        mean_vec_x+=torch.sum(dp.x,dim=0)
        std_vec_x+=torch.sum(dp.x**2,dim=0)
        num_accs_x+=dp.x.shape[0]

        mean_vec_edge+=torch.sum(dp.edge_attr,dim=0)
        std_vec_edge+=torch.sum(dp.edge_attr**2,dim=0)
        num_accs_edge+=dp.edge_attr.shape[0]

        mean_vec_y+=torch.sum(dp.y,dim=0)
        std_vec_y+=torch.sum(dp.y**2,dim=0)
        num_accs_y+=dp.y.shape[0]

        if(num_accs_x>max_accumulations or num_accs_edge>max_accumulations or num_accs_y>max_accumulations):
            break

    mean_vec_x = mean_vec_x/num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x/num_accs_x - mean_vec_x**2),eps)

    mean_vec_edge = mean_vec_edge/num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge/num_accs_edge - mean_vec_edge**2),eps)

    mean_vec_y = mean_vec_y/num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y/num_accs_y - mean_vec_y**2),eps)

    mean_std_list=[mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y]

    return mean_std_list

class GlobalAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = Linear(hidden_dim, hidden_dim)
        self.key = Linear(hidden_dim, hidden_dim)
        self.value = Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        
    def forward(self, x, batch=None):
        # If batch is None, treat all nodes as one graph
        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
            
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Global attention mechanism
        attention_logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_logits, dim=-1)
        return torch.matmul(attention_weights, value)

class MeshGraphNet(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, args, emb=False):
        super(MeshGraphNet, self).__init__()
        """
        MeshGraphNet model. This model is built upon Deepmind's 2021 paper.
        This model consists of three parts: (1) Preprocessing: encoder (2) Processor
        (3) postproccessing: decoder. Encoder has an edge and node decoders respectively.
        Processor has two processors for edge and node respectively. Note that edge attributes have to be
        updated first. Decoder is only for nodes.

        Input_dim: dynamic variables + node_type + node_position
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (1)

        """
        self.use_attention = True
        self.num_layers = args.num_layers
        self.hidden_dim = hidden_dim  # ADD THIS LINE - store hidden_dim as instance variable
        self.dropout_rate = getattr(args, 'dropout_rate', 0.38)  # Use args.dropout_rate if available
        self.operation_count = 0  # Add operation counter
        self.attention_freq = getattr(args, 'attention_freq', 
                                    max(1, self.num_layers // 2) if self.num_layers <= 8 else 8)

        # Add skip connection projection
        self.skip_projection = Linear(hidden_dim, hidden_dim)

        # encoder convert raw inputs into latent embeddings
        # self.node_encoder = Sequential(Linear(input_dim_node , hidden_dim),
        #                       ReLU(),
        #                       Linear( hidden_dim, hidden_dim),
        #                       LayerNorm(hidden_dim))
        self.node_encoder = Sequential(
            Linear(input_dim_node, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )
        # self.edge_encoder = Sequential(Linear( input_dim_edge , hidden_dim),
        #                       ReLU(),
        #                       Linear( hidden_dim, hidden_dim),
        #                       LayerNorm(hidden_dim)
        #                       )
        self.edge_encoder = Sequential(
            Linear(input_dim_edge, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )

        self.processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        self.global_attention = GlobalAttention(hidden_dim)

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim,hidden_dim))


        # decoder: only for node embeddings
        # self.decoder = Sequential(Linear( hidden_dim , hidden_dim),
        #                       ReLU(),
        #                       Linear( hidden_dim, output_dim)
        #                       )
        self.decoder = Sequential(
            Linear(hidden_dim, hidden_dim*2),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim*2, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, output_dim)
        )

    def build_processor_model(self):
        return ProcessorLayer

    def forward(self,data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = normalize(x,mean_vec_x,std_vec_x)
        edge_attr=normalize(edge_attr,mean_vec_edge,std_vec_edge)

        self.operation_count = 0

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension
        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # Step 2: perform message passing with latent node/edge embeddings
        layer_outputs = [x]  # Store layer outputs for skip connections
        for i in range(self.num_layers):
            # Count message passing operations
            num_edges = data.edge_index.shape[1]
            message_ops = num_edges * (self.hidden_dim ** 2)
            self.operation_count += message_ops
            # Add skip connection from 8 layers back (keep existing logic)
            if i >= 8 and i % 8 == 0:  # Every 8 layers after the 8th
                x = x + self.skip_projection(layer_outputs[i-8])
                
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)
            layer_outputs.append(x)  # Store current layer output
            
            # FIXED: Adaptive global attention frequency
            if (i + 1) % self.attention_freq == 0:
                num_nodes = x.size(0)
                attention_ops = num_nodes ** 2 * self.hidden_dim
                self.operation_count += attention_ops
                # Create batch index if not provided
                batch = getattr(data, 'batch', None)
                if batch is None:
                    batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
                    
                global_info = self.global_attention(x, batch=batch)
                x = x + 0.2 * global_info  # Mix with local representations

        # step 3: decode latent node embeddings into physical quantities of interest
        return self.decoder(x)
    
    def get_operation_count(self):
        return self.operation_count
    
    def loss(self, pred, inputs, mean_vec_y, std_vec_y):
        # In the new feature structure:
        # data.x[:, 0:2] = node positions (x, y)
        # data.x[:, 2] = is_hole_edge indicator
        # data.x[:, 3] = is_fixed indicator
        # data.x[:, 4] = is_displaced indicator
        # data.x[:, 5] = displacement_amount
        
        # Define which nodes to calculate loss for (not fixed nodes)
        # We'll calculate loss for nodes that are not fixed boundaries
        loss_mask = inputs.x[:, 3] < 0.5  # Only include nodes where is_fixed = 0
        
        # Normalize labels with dataset statistics
        labels = normalize(inputs.y, mean_vec_y, std_vec_y)
        
        # Ensure the shapes match
        if labels.shape != pred.shape:
            raise ValueError(f"Shape mismatch: labels shape {labels.shape} and pred shape {pred.shape} must match")

        # Find sum of square errors
        error = torch.sum((labels - pred) ** 2, axis=1)

        # Root and mean the errors for the nodes we calculate loss for
        loss = torch.sqrt(torch.mean(error[loss_mask]))

        return loss
    
class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(ProcessorLayer, self).__init__(  **kwargs )
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """
        self.attention = nn.Sequential(
            Linear(3*in_channels, 1),
            nn.Sigmoid()
        )

        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.edge_mlp = Sequential(Linear( 3* in_channels , out_channels),
                                   PReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))

        self.node_mlp = Sequential(Linear( 2* in_channels , out_channels),
                                   PReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))


        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size = None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shape [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """
        # Create mask before normalization
        # mask = (x.sum(dim=1) != 0)
        # print(f"Mask shape: {mask.shape}")
        # print(f"Number of real nodes: {mask.sum().item()}")

        # # print(f"Before propagate - x shape: {x.shape}")
        # print(f"Before propagate - edge_index shape: {edge_index.shape}")
        # print(f"Before propagate - edge_attr shape: {edge_attr.shape}")

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size) # out has the shape of [E, out_channels]

        # print(f"After propagate - x shape: {x.shape}")
        # print(f"After propagate - out shape: {out.shape}")

        ## Mask out the padded nodes
        # mask = (x.sum(dim=1) != 0)
        # print(f"Mask shape: {mask.shape}")
        # print(f"Number of real nodes: {mask.sum().item()}")

        # # Apply mask to filter out padded nodes
        # x = x[mask]
        # out = out[mask]

        # print(f"After Mask - x shape: {x.shape}")
        # print(f"After Mask - out shape: {out.shape}")

        updated_nodes = torch.cat([x, out], dim=1)        # Complete the aggregation through self-aggregation

        updated_nodes = x + self.node_mlp(updated_nodes) # residual connection

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        # """
        # print(f"message - x_i shape: {x_i.shape}")
        # print(f"message - x_j shape: {x_j.shape}")
        # print(f"message - edge_attr shape: {edge_attr.shape}")

        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1) # tmp_emb has the shape of [E, 3 * in_channels]
        updated_edges = self.edge_mlp(updated_edges) + edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size = None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """

        # The axis along which to index number of nodes.
        node_dim = 0

        # out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce='sum')
        # Ensure the output shape matches the input shape by specifying dim_size
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, dim_size=dim_size, reduce='sum')

        # print(f"node dim {node_dim}")
        # print(f"aggregate - updated_edges shape: {updated_edges.shape}")
        # print(f"aggregate - edge_index shape: {edge_index.shape}")
        # print(f"aggregate - out shape after scatter: {out.shape}")

        return out, updated_edges
    
def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    elif args.opt_scheduler == 'cosine':
    # Cosine annealing with warmup
        def lr_lambda(current_step: int):
            warmup_steps = 200
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, args.epochs - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return scheduler, optimizer

def analyze_node_features(dataset):
    """Analyze the structure of node features in the dataset"""
    sample_data = dataset[0]
    
    print("\n===== NODE FEATURE ANALYSIS =====")
    print(f"Shape of data.x: {sample_data.x.shape}")
    
    # Node positions (x, y)
    positions = sample_data.x[:, :2]
    print("\nNode positions (x, y):")
    print(f"  Shape: {positions.shape}")
    print(f"  Mean: {torch.mean(positions, dim=0)}")
    print(f"  Min: {torch.min(positions, dim=0)[0]}")
    print(f"  Max: {torch.max(positions, dim=0)[0]}")
    
    # Hole edge indicator
    is_hole_edge = sample_data.x[:, 2]
    print("\nHole edge indicator:")
    print(f"  Shape: {is_hole_edge.shape}")
    print(f"  Number of hole edge nodes: {torch.sum(is_hole_edge > 0.5).item()}")
    
    # Fixed nodes indicator
    is_fixed = sample_data.x[:, 3]
    print("\nFixed nodes indicator:")
    print(f"  Shape: {is_fixed.shape}")
    print(f"  Number of fixed nodes: {torch.sum(is_fixed > 0.5).item()}")
    
    # Displaced nodes indicator
    is_displaced = sample_data.x[:, 4]
    print("\nDisplaced nodes indicator:")
    print(f"  Shape: {is_displaced.shape}")
    print(f"  Number of displaced nodes: {torch.sum(is_displaced > 0.5).item()}")
    
    # Displacement amount
    displacement_amount = sample_data.x[:, 5]
    print("\nDisplacement amount:")
    print(f"  Shape: {displacement_amount.shape}")
    print(f"  Average displacement (for displaced nodes): {torch.sum(displacement_amount * is_displaced) / torch.sum(is_displaced)}")
    
    print("=================================\n")

# def train(dataset, device, stats_list, args):
#     '''
#     Performs a training loop on the dataset for MeshGraphNets. Also calls
#     test and validation functions.
#     '''

#     df = pd.DataFrame(columns=['epoch','train_loss','test_loss', 'velo_val_loss'])

#     # Define the model name for saving
#     model_name = 'model_nl' + str(args.num_layers) + '_bs' + str(args.batch_size) + \
#                  '_hd' + str(args.hidden_dim) + '_ep' + str(args.epochs) + '_wd' + str(args.weight_decay) + \
#                  '_lr' + str(args.lr) + '_shuff_' + str(args.shuffle) + '_tr' + str(args.train_size) + '_te' + str(args.test_size)

#     # torch_geometric DataLoaders are used for handling the data of lists of graphs
#     loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
#     test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)

#     # The statistics of the data are decomposed
#     [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
#     (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
#         mean_vec_x.to(device), std_vec_x.to(device), mean_vec_edge.to(device), std_vec_edge.to(device), mean_vec_y.to(device), std_vec_y.to(device))

#     # Build model
#     num_node_features = dataset[0].x.shape[1]
#     num_edge_features = dataset[0].edge_attr.shape[1]
#     num_classes = 1

#     model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes, args).to(device)
#     scheduler, opt = build_optimizer(args, model.parameters())

#     # Train
#     losses = []
#     test_losses = []
#     velo_val_losses = []
#     best_test_loss = np.inf
#     best_model = None

#     for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
#         total_loss = 0
#         model.train()
#         num_loops = 0
#         for batch in loader:
#             batch = batch.to(device)
#             opt.zero_grad()  # zero gradients each time
#             pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
#             loss = model.loss(pred, batch, mean_vec_y, std_vec_y)
#             loss.backward()  # backpropagate loss
#             opt.step()
#             total_loss += loss.item()
#             num_loops += 1
#         total_loss /= num_loops
#         losses.append(total_loss)

#         # Every tenth epoch, calculate acceleration test loss and velocity validation loss
#         if epoch % 10 == 0:
#             if args.save_velo_val:
#                 # Save velocity evaluation
#                 test_loss, velo_val_rmse = test(test_loader, device, model, mean_vec_x, std_vec_x, mean_vec_edge,
#                                                 std_vec_edge, mean_vec_y, std_vec_y, args.save_velo_val)
#                 velo_val_losses.append(velo_val_rmse.item())
#             else:
#                 test_loss, _ = test(test_loader, device, model, mean_vec_x, std_vec_x, mean_vec_edge,
#                                     std_vec_edge, mean_vec_y, std_vec_y, args.save_velo_val)

#             test_losses.append(test_loss.item())

#             # Saving model
#             if not os.path.isdir(args.checkpoint_dir):
#                 os.mkdir(args.checkpoint_dir)

#             PATH = os.path.join(args.checkpoint_dir, model_name + '.csv')
#             df.to_csv(PATH, index=False)

#             # Save the model if the current one is better than the previous best
#             if test_loss < best_test_loss:
#                 best_test_loss = test_loss
#                 best_model = copy.deepcopy(model)

#         else:
#             # If not the tenth epoch, append the previously calculated loss to the
#             # list in order to be able to plot it on the same plot as the training losses
#             if args.save_velo_val:
#                 test_losses.append(test_losses[-1])
#                 velo_val_losses.append(velo_val_losses[-1])

#         new_row = pd.DataFrame({'epoch': [epoch], 'train_loss': [losses[-1]], 'test_loss': [test_losses[-1]]})
#         if args.save_velo_val:
#             new_row['velo_val_loss'] = velo_val_losses[-1]
#         df = pd.concat([df, new_row], ignore_index=True)

#         if epoch % 100 == 0:
#             if args.save_velo_val:
#                 print("train loss", str(round(total_loss, 2)),
#                       "test loss", str(round(test_loss.item(), 2)),
#                       "velo loss", str(round(velo_val_rmse.item(), 5)))
#             else:
#                 print("train loss", str(round(total_loss, 2)), "test loss", str(round(test_loss.item(), 2)))

#             if args.save_best_model:
#                 PATH = os.path.join(args.checkpoint_dir, model_name + '.pt')
#                 torch.save(best_model.state_dict(), PATH)

#     # Plot comparison between predicted, ground truth, and error for the last step
#     plot_name = 'x_stress_last_epoch'
#     visualize(test_loader, best_model, postprocess_dir, plot_name, stats_list, sample_index=0)
    
#     return test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader

def test(loader, device, test_model, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, is_validation, delta_t=0.01, save_model_preds=False, model_type=None):
    '''
    Calculates test set losses and validation set errors.
    '''
    loss = 0
    velo_rmse = 0
    num_loops = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            # Calculate the loss for the model given the test set
            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss += test_model.loss(pred, data, mean_vec_y, std_vec_y)

            # Calculate validation error if asked to
            if is_validation:
                # Unnormalize the predictions and ground truth
                eval_y = unnormalize(pred, mean_vec_y, std_vec_y)
                gs_y = data.y

                # Calculate the error
                error = torch.sum((eval_y - gs_y) ** 2, axis=1)
                velo_rmse += torch.sqrt(torch.mean(error))

        num_loops += 1

    return loss / num_loops, velo_rmse / num_loops

def train(train_dataset, val_dataset, device, stats_list, args):
    '''
    Performs a training loop on the dataset for MeshGraphNets with proper validation.
    '''
    df = pd.DataFrame(columns=['epoch','train_loss','val_loss', 'velo_val_loss'])

    # Define the model name for saving
    model_name = 'model_nl' + str(args.num_layers) + '_bs' + str(args.batch_size) + \
                 '_hd' + str(args.hidden_dim) + '_ep' + str(args.epochs) + '_wd' + str(args.weight_decay) + \
                 '_lr' + str(args.lr) + '_shuff_' + str(args.shuffle)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # The statistics of the data are decomposed
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(device), std_vec_x.to(device), mean_vec_edge.to(device), 
        std_vec_edge.to(device), mean_vec_y.to(device), std_vec_y.to(device))

    # Build model
    num_node_features = train_dataset[0].x.shape[1]
    num_edge_features = train_dataset[0].edge_attr.shape[1]
    num_classes = 1

    model = MeshGraphNet(num_node_features, num_edge_features, 
                        args.hidden_dim, num_classes, args).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # Train
    losses = []
    val_losses = []
    velo_val_losses = []
    best_val_loss = float('inf')
    best_model = None
    patience = 1500  # Early stopping patience
    patience_counter = 0

    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        # Training step
        model.train()
        total_loss = 0
        num_loops = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss = model.loss(pred, batch, mean_vec_y, std_vec_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent gradient explosion
            opt.step()
            total_loss += loss.item()
            num_loops += 1
            
        if scheduler is not None:
            scheduler.step()
            
        train_loss = total_loss / num_loops
        losses.append(train_loss)
        # Validation step
        model.eval()
        val_loss, velo_val_rmse = test(val_loader, device, model, mean_vec_x, std_vec_x, 
                                      mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, 
                                      args.save_velo_val)
        
        val_losses.append(val_loss.item())
        if args.save_velo_val:
            velo_val_losses.append(velo_val_rmse.item())
            
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        # Logging
        print(f"Epoch {epoch}: Train Loss = {train_loss:.5f}, Val Loss = {val_loss:.5f}")
        
        # Save progress
        new_row = pd.DataFrame({
            'epoch': [epoch], 
            'train_loss': [train_loss], 
            'val_loss': [val_loss.item()]
        })
        if args.save_velo_val:
            new_row['velo_val_loss'] = velo_val_rmse.item()
        df = pd.concat([df, new_row], ignore_index=True)
        
        if args.save_best_model and epoch % 5 == 0:
            PATH = os.path.join(checkpoint_dir, model_name + '.pt')
            torch.save(best_model.state_dict(), PATH)
    
    # Save final dataframe
    PATH = os.path.join(checkpoint_dir, model_name + '.csv')
    df.to_csv(PATH, index=False)
    
    # Plot results from final model
    plot_name = 'final_epoch_results'
    visualize(val_loader, best_model, postprocess_dir, plot_name, stats_list)
    
    return val_losses, losses, velo_val_losses, best_model


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
    print(f"\n{'='*60}")
    print("INFERENCE TIME BENCHMARK")
    print(f"{'='*60}")
    
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
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            prediction = model(sample, *stats_gpu)
            
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    # Calculate average inference time
    total_inference_time = end_time - start_time
    avg_inference_time = total_inference_time / num_runs
    
    # Estimate FEM simulation time (you should replace this with actual FEM timing)
    # These are typical values - you should measure actual FEM times for your meshes
    estimated_fem_times = {
        'simple_mesh': 300,      # 5 minutes for simple mesh
        'medium_mesh': 1800,     # 30 minutes for medium complexity
        'complex_mesh': 7200,    # 2 hours for complex mesh
        'your_mesh': 600        # Estimate for your specific mesh - UPDATE THIS
    }
    
    results = {
        'gnn_inference_time': avg_inference_time,
        'num_runs': num_runs,
        'total_time': total_inference_time,
        'speedup_analysis': {}
    }
    
    print(f"GNN Inference Results:")
    print(f"  Average inference time: {avg_inference_time:.6f} seconds")
    print(f"  Total time for {num_runs} runs: {total_inference_time:.3f} seconds")
    print(f"  Inference frequency: {1/avg_inference_time:.1f} predictions/second")
    
    print(f"\nSpeedup Analysis vs FEM:")
    print(f"{'FEM Type':<15} {'FEM Time (s)':<15} {'Speedup':<15} {'Time Saved':<15}")
    print("-" * 65)
    
    for fem_type, fem_time in estimated_fem_times.items():
        speedup = fem_time / avg_inference_time
        time_saved = fem_time - avg_inference_time
        results['speedup_analysis'][fem_type] = {
            'fem_time': fem_time,
            'speedup': speedup,
            'time_saved': time_saved
        }
        print(f"{fem_type:<15} {fem_time:<15.1f} {speedup:<15.0f}× {time_saved:<15.1f}s")
    
    # Break-even analysis for training cost
    training_time_estimate = 3600  # 1 hour estimate - update with your actual training time
    print(f"\nBreak-even Analysis:")
    print(f"Training time estimate: {training_time_estimate:.0f} seconds")
    
    for fem_type, analysis in results['speedup_analysis'].items():
        time_saved_per_pred = analysis['time_saved']
        break_even_predictions = training_time_estimate / time_saved_per_pred if time_saved_per_pred > 0 else float('inf')
        print(f"  {fem_type}: Break-even after {break_even_predictions:.0f} predictions")
    
    return results

def save_speedup_analysis(results, results_folder):
    """Save speedup analysis results to files."""
    import json
    
    # Save as JSON
    json_path = os.path.join(results_folder, 'speedup_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Speedup comparison
    fem_types = list(results['speedup_analysis'].keys())
    speedups = [results['speedup_analysis'][ft]['speedup'] for ft in fem_types]
    
    bars = ax1.bar(fem_types, speedups, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Speedup Factor (×)')
    ax1.set_title('GNN vs FEM Speedup')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{speedup:.0f}×', ha='center', va='bottom')
    
    # Plot 2: Time comparison
    fem_times = [results['speedup_analysis'][ft]['fem_time'] for ft in fem_types]
    gnn_time = results['gnn_inference_time']
    
    x = range(len(fem_types))
    ax2.bar([i - 0.2 for i in x], fem_times, 0.4, label='FEM', color='red', alpha=0.7)
    ax2.bar([i + 0.2 for i in x], [gnn_time] * len(fem_types), 0.4, 
           label='GNN', color='blue', alpha=0.7)
    
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Absolute Time Comparison')
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(fem_types)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(results_folder, 'speedup_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSpeedup analysis saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Plot: {plot_path}")


def save_plots(args, train_losses, val_losses, test_losses=None, velo_val_losses=None):
    model_name = 'model_nl' + str(args.num_layers) + '_bs' + str(args.batch_size) + \
                 '_hd' + str(args.hidden_dim) + '_ep' + str(args.epochs) + '_wd' + str(args.weight_decay) + \
                 '_lr' + str(args.lr) + '_shuff_' + str(args.shuffle) + '_tr' + str(args.train_size) + '_te' + str(args.test_size)

    if not os.path.isdir(postprocess_dir):
        os.mkdir(postprocess_dir)

    PATH = os.path.join(postprocess_dir, model_name + '.pdf')

    f = plt.figure(figsize=(10, 6))
    plt.title('Losses Plot')
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    
    if test_losses is not None and len(test_losses) > 0:
        # If we only have final test loss, show it as a point
        if len(test_losses) == 1:
            plt.scatter(len(train_losses)-1, test_losses[0], color='green', label="Final Test loss", s=100, zorder=5)
        else:
            plt.plot(test_losses, label="Test loss")
    
    # Removed the velocity loss plotting code
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PATH, bbox_inches='tight')
    plt.show()
    print(f"Plot saved at: {PATH}")

def evaluate_final_model(test_dataset, best_model, device, stats_list, args):
    """
    Evaluate the final model on the test set after training is complete.
    """
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(device), std_vec_x.to(device), mean_vec_edge.to(device), 
        std_vec_edge.to(device), mean_vec_y.to(device), std_vec_y.to(device))
    
    test_loss, test_rmse = test(test_loader, device, best_model, mean_vec_x, std_vec_x, 
                              mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, True)
    
    print(f"Final Test Loss: {test_loss:.5f}")
    print(f"Final Test RMSE: {test_rmse:.5f}")
    
    # Generate visualization on test set
    plot_name = 'test_set_final_results'
    visualize(test_loader, best_model, postprocess_dir, plot_name, stats_list)
    
    return test_loss, test_rmse

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

# for args in [
#         {
#          'model_type': 'meshgraphnet',
#          'num_layers': 24,
#          'batch_size': 8,   #originally 16
#          'hidden_dim': 64,
#          'epochs': 2700,      # originally 5000
#          'opt': 'adam',
#          'opt_scheduler': 'cosine',
#          'opt_decay_step': 75,    # Add this parameter
#          'opt_decay_rate': 0.83,    # Add this parameter
#          'opt_restart': 0,
#          'weight_decay': 1.17e-6,
#          'lr': 0.0005359,
#          'train_size': 400,  #originally 45
#          'test_size': 100,   #originally 10
#          'device':'cuda',
#          'shuffle': True,
#          'save_velo_val': True,
#          'save_best_model': True,
#          'checkpoint_dir': './best_models/',
#          'postprocess_dir': './2d_loss_plots/'},
#     ]:
#         args = objectview(args)

for args in [
        {
        'model_type': 'meshgraphnet',
        'num_layers': 6,  # Updated from trial.params to optimal value
        'batch_size': 4,  # Updated from trial.params to optimal value
        'hidden_dim': 48,  # Updated from trial.params to optimal value
        'dropout_rate': 0.253,  # Updated from trial.params to optimal value
        'attention_freq': 999,  # Updated from trial.params to optimal value
        'epochs': 500,
        'opt': 'adam',  # Updated from trial.params to optimal value
        'opt_scheduler': 'step',
        'opt_decay_step': 46,  # Updated from trial.params to optimal value
        'opt_decay_rate': 0.668,  # Updated from trial.params to optimal value
        'opt_restart': 0,
        'weight_decay': 1.07e-05,  # Updated from trial.params to optimal value
        'lr': 8.24e-04,  # Updated from trial.params to optimal value
        'train_size': 400,  #originally 45
        'test_size': 100,   #originally 10
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'shuffle': True,
        'save_velo_val': True,
        'save_best_model': True,
        'checkpoint_dir': './best_models/',
        'postprocess_dir': './2d_loss_plots/'},
    ]:
        args = objectview(args)


#To ensure reproducibility the best we can, here we control the sources of
#randomness by seeding the various random number generators used in this Colab
#For more information, see: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(5)  #Torch
random.seed(5)        #Python
np.random.seed(5)     #NumPy

dataset = torch.load(file_path, weights_only=False)[:(args.train_size+args.test_size)]

sample_data = dataset[0]  # Get first graph in dataset

print("Node feature tensor shape:", sample_data.x.shape)
print("Number of nodes:", sample_data.x.shape[0])
print("Node feature dimension:", sample_data.x.shape[1])

# Print statistics
print("Node features min/max/mean:", 
      torch.min(sample_data.x).item(), 
      torch.max(sample_data.x).item(),
      torch.mean(sample_data.x).item())

# Examine the first few nodes
print("First 3 node features:")
for i in range(min(3, sample_data.x.shape[0])):
    print(f"Node {i}:", sample_data.x[i])

# If you suspect there's a specific structure (like first N features are position, next M are one-hot encoded type)
# Try to verify by checking patterns:
print("\nExample node types (if one-hot encoded):")
if sample_data.x.shape[1] > 3:  # Assuming at least a few features
    # Look at potential one-hot encoded section (often in latter part of feature vector)
    potential_onehot = sample_data.x[:5, 2:]  # First 5 nodes, features from 3rd onward
    print(potential_onehot)
    
    # Check if any rows sum to 1 (typical of one-hot encoding)
    row_sums = torch.sum(potential_onehot, dim=1)
    print("Sum of potential one-hot section:", row_sums)

analyze_node_features(dataset)

# Calculate split sizes
total_size = len(dataset)
train_size = int(total_size * 0.7)  # 70% training
val_size = int(total_size * 0.15)   # 15% validation 
test_size = total_size - train_size - val_size  # 15% testing

print(f"Dataset size: {total_size}")
print(f"Training set size: {train_size}")
print(f"Validation set size: {val_size}")
print(f"Test set size: {test_size}")

# Create the splits
if args.shuffle:
    random.shuffle(dataset)

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size+val_size]
test_dataset = dataset[train_size+val_size:]

# Update args
args.train_size = train_size
args.val_size = val_size
args.test_size = test_size

# Get statistics for normalization
stats_list = get_stats(dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
print(device)


# Add after your existing dataset preparation but before training
print("Starting timing benchmark...")

# Run timing benchmark
timing_results = benchmark_attention_frequencies(
    dataset=dataset, 
    device=device, 
    stats_list=stats_list, 
    base_args=args,
    num_epochs=3  # Use fewer epochs for quick testing
)

# Analyze and save results
timing_df = analyze_timing_results(timing_results)
save_timing_results(timing_df, results_folder)

# Now continue with your regular training...
val_losses, losses, velo_val_losses, best_model = train(
    train_dataset, val_dataset, device, stats_list, args
)

# val_losses, losses, velo_val_losses, best_model = train(
#     train_dataset, val_dataset, device, stats_list, args
# )

# # Final evaluation on test set
# final_test_loss, final_test_rmse = evaluate_final_model(test_dataset, best_model, device, stats_list, args)
# print(f"Final Test Loss: {final_test_loss:.5f}")
# print(f"Final Test RMSE: {final_test_rmse:.5f}")

# # Plot losses
# test_losses = [final_test_loss.item()]  # Just use the final test loss
# save_plots(args, losses, val_losses, test_losses)