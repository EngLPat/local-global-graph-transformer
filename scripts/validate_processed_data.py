"""
FCLGA GraphTransformer - Data Validation Script

Validates and visualizes processed data to ensure pipeline worked correctly.
Creates plots of mesh geometry and strain fields.

Usage:
    python scripts/validate_processed_data.py
    python scripts/validate_processed_data.py --sample 0 --num-samples 3
"""

import argparse
import sys
from pathlib import Path
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Paths
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_PLOTS = PROJECT_ROOT / "results" / "plots" / "validation"

# Create output directory
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all processed data files."""
    print(f"\n{'='*60}")
    print("LOADING PROCESSED DATA")
    print(f"{'='*60}\n")
    
    data = {}
    
    # Load node GNN data
    node_data_path = DATA_PROCESSED / "node_gnn_data.pt"
    if node_data_path.exists():
        data['node_gnn_data'] = torch.load(node_data_path, weights_only=False)
        print(f"✓ Loaded node_gnn_data.pt")
        print(f"  - Number of samples: {len(data['node_gnn_data'])}")
        if len(data['node_gnn_data']) > 0:
            sample = data['node_gnn_data'][0]
            print(f"  - Sample 0 nodes: {sample.x.shape[0]}")
            print(f"  - Sample 0 edges: {sample.edge_index.shape[1]}")
            print(f"  - Node features shape: {sample.x.shape}")
            # Check if node IDs are stored
            if hasattr(sample, 'node_ids'):
                print(f"  - Node IDs available: Yes")
            else:
                print(f"  - Node IDs available: No (may cause ordering issues!)")
    else:
        print(f"✗ node_gnn_data.pt not found at {node_data_path}")
    
    # Load triangulation data
    tri_data_path = DATA_PROCESSED / "triangulation_data.pkl"
    if tri_data_path.exists():
        with open(tri_data_path, 'rb') as f:
            data['triangulation_data'] = pickle.load(f)
        print(f"✓ Loaded triangulation_data.pkl")
        print(f"  - Number of samples: {len(data['triangulation_data'])}")
    else:
        print(f"✗ triangulation_data.pkl not found at {tri_data_path}")
    
    # Load strains tensor
    strains_path = DATA_PROCESSED / "strains.pt"
    if strains_path.exists():
        data['strains'] = torch.load(strains_path)
        print(f"✓ Loaded strains.pt")
        print(f"  - Tensor shape: {data['strains'].shape}")
        print(f"  - [num_samples, max_nodes, features]")
        print(f"  - Strain range: [{data['strains'].min():.6f}, {data['strains'].max():.6f}]")
    else:
        print(f"✗ strains.pt not found at {strains_path}")
    
    # Load geometry data (optional)
    geom_path = DATA_PROCESSED / "plate_geometry_data.pt"
    geom_pkl_path = DATA_PROCESSED / "plate_geometry_data.pkl"
    if geom_path.exists():
        data['geometry'] = torch.load(geom_path)
        print(f"✓ Loaded plate_geometry_data.pt")
        print(f"  - Shape: {data['geometry'].shape}")
        print(f"  - [num_samples, 6] = [L, W, hole_radius, hole_x, hole_y, displacement]")
    elif geom_pkl_path.exists():
        with open(geom_pkl_path, 'rb') as f:
            data['geometry'] = pickle.load(f)
        print(f"✓ Loaded plate_geometry_data.pkl")
        print(f"  - Number of samples: {len(data['geometry'])}")
    else:
        print(f"⚠ No geometry data found (optional)")
    
    print(f"\n{'='*60}\n")
    
    return data


def plot_mesh_geometry(data, sample_idx, save_path):
    """Plot the mesh geometry for a sample."""
    node_data = data['node_gnn_data'][sample_idx]
    
    # Extract node positions (x, y coordinates)
    positions = node_data.x.numpy()
    
    # Get actual number of nodes (excluding padding)
    if hasattr(node_data, 'node_ids'):
        num_actual_nodes = len(node_data.node_ids)
    else:
        num_actual_nodes = positions.shape[0]
    
    x_coords = positions[:num_actual_nodes, 0]
    y_coords = positions[:num_actual_nodes, 1]
    
    # Extract edges
    edge_index = node_data.edge_index.numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Mesh with edges
    ax1.scatter(x_coords, y_coords, c='blue', s=10, alpha=0.6, label='Nodes')
    
    # Draw edges
    for i in range(0, edge_index.shape[1], 10):  # Plot every 10th edge to avoid clutter
        node_start, node_end = edge_index[:, i]
        if node_start < len(x_coords) and node_end < len(x_coords):
            ax1.plot([x_coords[node_start], x_coords[node_end]], 
                    [y_coords[node_start], y_coords[node_end]], 
                    'r-', alpha=0.1, linewidth=0.5)
    
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.set_title(f'Sample {sample_idx}: Mesh Geometry\n{len(x_coords)} nodes, {edge_index.shape[1]} edges', 
                  fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Node density heatmap
    hist = ax2.hist2d(x_coords, y_coords, bins=50, cmap='YlOrRd')
    plt.colorbar(hist[3], ax=ax2, label='Node Density')
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)
    ax2.set_title(f'Sample {sample_idx}: Node Density Distribution', 
                  fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved mesh geometry plot: {save_path.name}")


def plot_strain_field(data, sample_idx, save_path):
    """Plot the strain field on the mesh using triangulation."""
    # Get node positions
    node_data = data['node_gnn_data'][sample_idx]
    positions = node_data.x.numpy()
    
    # Get actual number of nodes (excluding padding)
    if hasattr(node_data, 'node_ids'):
        num_actual_nodes = len(node_data.node_ids)
    else:
        num_actual_nodes = positions.shape[0]
    
    x_coords = positions[:num_actual_nodes, 0]
    y_coords = positions[:num_actual_nodes, 1]
    
    # Get strains for this sample
    strains = data['strains'][sample_idx].numpy().flatten()
    valid_strains = strains[:num_actual_nodes]
    
    # Get triangulation
    triangulation = tri.Triangulation(x_coords, y_coords)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Strain field with filled contours
    levels = np.linspace(valid_strains.min(), valid_strains.max(), 20)
    contour = ax1.tricontourf(triangulation, valid_strains, levels=levels, cmap='RdYlBu_r')
    cbar1 = plt.colorbar(contour, ax=ax1, label='E11 Strain')
    ax1.triplot(triangulation, 'k-', alpha=0.1, linewidth=0.3)
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.set_title(f'Sample {sample_idx}: E11 Strain Field (Contour)\nMin: {valid_strains.min():.6f}, Max: {valid_strains.max():.6f}', 
                  fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    
    # Plot 2: Strain field with scatter points
    scatter = ax2.scatter(x_coords, y_coords, c=valid_strains, 
                         cmap='RdYlBu_r', s=20, edgecolors='none')
    cbar2 = plt.colorbar(scatter, ax=ax2, label='E11 Strain')
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)
    ax2.set_title(f'Sample {sample_idx}: E11 Strain Distribution (Nodes)', 
                  fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved strain field plot: {save_path.name}")


def plot_statistics(data, save_path):
    """Plot overall statistics of the dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of nodes per sample
    node_counts = [d.x.shape[0] for d in data['node_gnn_data']]
    axes[0, 0].hist(node_counts, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Nodes', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title(f'Node Count Distribution\nMean: {np.mean(node_counts):.0f}, Std: {np.std(node_counts):.0f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Number of edges per sample
    edge_counts = [d.edge_index.shape[1] for d in data['node_gnn_data']]
    axes[0, 1].hist(edge_counts, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Edges', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title(f'Edge Count Distribution\nMean: {np.mean(edge_counts):.0f}, Std: {np.std(edge_counts):.0f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Strain value distribution
    all_strains = data['strains'].numpy().flatten()
    non_zero_strains = all_strains[all_strains != 0]
    axes[1, 0].hist(non_zero_strains, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('E11 Strain Value', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title(f'Strain Value Distribution\nMean: {non_zero_strains.mean():.6f}, Std: {non_zero_strains.std():.6f}', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Geometry parameters (if available)
    if 'geometry' in data:
        geom = data['geometry']
        if isinstance(geom, torch.Tensor):
            geom = geom.numpy()
        else:
            geom = np.array(geom)
        
        # Plot displacement values (last column)
        displacements = geom[:, -1] if geom.ndim > 1 else [g[-1] for g in geom]
        axes[1, 1].hist(displacements, bins=20, color='purple', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Displacement Value', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title(f'Applied Displacement Distribution\nRange: [{min(displacements):.2f}, {max(displacements):.2f}]', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No geometry data available', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')
    
    plt.suptitle('Dataset Statistics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved statistics plot: {save_path.name}")


def plot_edge_connectivity(data, sample_idx, save_path):
    """
    Plot edge connectivity showing 3 scenarios to detect off-by-one errors:
    1. Edges as stored
    2. Edges with indices + 1
    3. Edges with indices - 1
    """
    node_data = data['node_gnn_data'][sample_idx]
    positions = node_data.x.numpy()
    edge_index = node_data.edge_index.numpy()
    
    num_actual_nodes = positions.shape[0]
    x_coords, y_coords = positions[:, 0], positions[:, 1]
    
    # Check for issues
    issues = []
    connected_nodes = np.unique(edge_index.flatten())
    isolated = set(range(num_actual_nodes)) - set(connected_nodes)
    if isolated:
        issues.append(f"⚠ {len(isolated)} isolated nodes (expected for padding)")
    
    # Check off-by-one scenarios
    edge_plus_one = edge_index + 1
    edge_minus_one = edge_index - 1
    
    valid_normal = edge_index.shape[1]
    valid_plus_one = np.sum((edge_plus_one[0] < num_actual_nodes) & (edge_plus_one[1] < num_actual_nodes))
    valid_minus_one = np.sum((edge_minus_one[0] >= 0) & (edge_minus_one[1] >= 0) & 
                            (edge_minus_one[0] < num_actual_nodes) & (edge_minus_one[1] < num_actual_nodes))
    
    # Create 3x2 plot
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle(f'Edge Connectivity Analysis - Sample {sample_idx}', fontsize=16, fontweight='bold')
    
    # Show ALL edges (no sampling) - may be dense but shows full connectivity
    edge_sample_rate = 1  # Changed from sampling to show all edges
    
    # Row 1: AS STORED
    ax = axes[0, 0]
    ax.scatter(x_coords, y_coords, c='blue', s=15, alpha=0.5, label='Nodes')
    for i in range(0, edge_index.shape[1], edge_sample_rate):
        src, dst = edge_index[:, i]
        if src < num_actual_nodes and dst < num_actual_nodes:
            ax.plot([x_coords[src], x_coords[dst]], [y_coords[src], y_coords[dst]], 
                   'gray', alpha=0.1, linewidth=0.5)
    ax.set_title(f'EDGES AS STORED\\n{valid_normal} edges (ALL SHOWN)', fontweight='bold', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    # Row 1: +1 SHIFT
    ax = axes[0, 1]
    ax.scatter(x_coords, y_coords, c='blue', s=15, alpha=0.5, label='Nodes')
    for i in range(0, edge_plus_one.shape[1], edge_sample_rate):
        src, dst = edge_plus_one[:, i]
        if src < num_actual_nodes and dst < num_actual_nodes:
            ax.plot([x_coords[src], x_coords[dst]], [y_coords[src], y_coords[dst]], 
                   'red', alpha=0.1, linewidth=0.5)
    ax.set_title(f'EDGES WITH INDICES + 1\n{valid_plus_one} valid / {edge_plus_one.shape[1]} total', 
                fontweight='bold', fontsize=12, color='darkred')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    # Row 2: -1 SHIFT
    ax = axes[1, 0]
    ax.scatter(x_coords, y_coords, c='blue', s=15, alpha=0.5, label='Nodes')
    for i in range(0, edge_minus_one.shape[1], edge_sample_rate):
        src, dst = edge_minus_one[:, i]
        if src >= 0 and dst >= 0 and src < num_actual_nodes and dst < num_actual_nodes:
            ax.plot([x_coords[src], x_coords[dst]], [y_coords[src], y_coords[dst]], 
                   'green', alpha=0.1, linewidth=0.5)
    ax.set_title(f'EDGES WITH INDICES - 1\n{valid_minus_one} valid / {edge_minus_one.shape[1]} total', 
                fontweight='bold', fontsize=12, color='darkgreen')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    # Row 2: OFF-BY-ONE ANALYSIS
    ax = axes[1, 1]
    ax.axis('off')
    analysis_text = f"""OFF-BY-ONE ERROR ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sample {sample_idx}
Nodes: {num_actual_nodes}

SCENARIO COMPARISON:
▸ AS STORED (gray):
  Valid: {valid_normal}/{edge_index.shape[1]}
  
▸ INDICES + 1 (red):
  Valid: {valid_plus_one}/{edge_plus_one.shape[1]}
  Out-of-bounds: {edge_plus_one.shape[1] - valid_plus_one}
  
▸ INDICES - 1 (green):
  Valid: {valid_minus_one}/{edge_minus_one.shape[1]}
  Invalid: {edge_minus_one.shape[1] - valid_minus_one}

INTERPRETATION:
"""
    if valid_plus_one < edge_index.shape[1] * 0.9:
        analysis_text += "✓ +1 shift breaks connectivity\n"
    else:
        analysis_text += "⚠ +1 shift looks valid!\n"
    
    if valid_minus_one < edge_index.shape[1] * 0.9:
        analysis_text += "✓ -1 shift breaks connectivity\n"
    else:
        analysis_text += "⚠ -1 shift looks valid!\n"
    
    if valid_plus_one < edge_index.shape[1] * 0.9 and valid_minus_one < edge_index.shape[1] * 0.9:
        analysis_text += "\n✓ AS STORED is correct\n"
    else:
        analysis_text += "\n⚠ Possible off-by-one error!\n"
    
    ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Row 3: DIAGNOSTICS
    ax = axes[2, 0]
    ax.axis('off')
    stats_text = f"""GRAPH STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nodes: {num_actual_nodes}
Edges: {edge_index.shape[1]}
Isolated nodes: {len(isolated)}
(padding expected)

Edge index range:
Min: {edge_index.min()}
Max: {edge_index.max()}
Expected: [0, {num_actual_nodes-1}]

NOTES:
• Isolated padded nodes are OK
• Training should mask them:
  mask = (x.sum(dim=1) != 0)
• ALL edges shown in plots above
  (not sampled - full connectivity)
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Row 3: ISSUES
    ax = axes[2, 1]
    ax.axis('off')
    issues_text = "DETECTED ISSUES\n" + "━"*30 + "\n\n"
    if issues:
        for issue in issues:
            issues_text += f"{issue}\n"
    else:
        issues_text += "✓ No critical issues\n"
    
    ax.text(0.05, 0.95, issues_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved edge connectivity plot: {save_path.name}")
    return issues


def compare_with_legacy_dataset(data):
    """Compare current dataset structure with expected legacy format."""
    print(f"\n{'='*60}")
    print("COMPARING WITH LEGACY FORMAT")
    print(f"{'='*60}\n")
    
    sample = data['node_gnn_data'][0]
    
    print("Checking graph structure:")
    print(f"  • Node features: {sample.x.shape}")
    print(f"    Expected: [num_nodes, 6] = [x, y, is_hole_edge, is_fixed, is_displaced, displacement]")
    
    print(f"  • Edge index: {sample.edge_index.shape}")
    print(f"    Expected: [2, num_edges]")
    print(f"    Format: edge_index[0] = source nodes, edge_index[1] = target nodes")
    
    if hasattr(sample, 'edge_attr'):
        print(f"  • Edge attributes: {sample.edge_attr.shape}")
        print(f"    Expected: [num_edges, 3] = [dx, dy, distance]")
    else:
        print(f"  • Edge attributes: NOT FOUND")
    
    print(f"  • Target (y): {sample.y.shape}")
    print(f"    Expected: [num_nodes, 1] = strain values")
    
    # Check if graph is directed or undirected
    edge_index = sample.edge_index.numpy()
    forward = set(tuple(edge_index[:, i]) for i in range(edge_index.shape[1]))
    reverse = set((dst, src) for src, dst in forward)
    bidirectional = len(forward & reverse) / len(forward)
    
    print(f"\n  • Graph type: {'Undirected' if bidirectional > 0.9 else 'Directed'}")
    print(f"    Bidirectional edges: {bidirectional:.1%}")
    
    # Check for issues that cause homogeneous predictions
    print(f"\nChecking for training issues:")
    
    # Issue 1: All edges point to wrong nodes
    if edge_index.max() >= sample.x.shape[0]:
        print(f"  ✗ CRITICAL: Edge indices exceed number of nodes!")
        print(f"    Max edge index: {edge_index.max()}, Num nodes: {sample.x.shape[0]}")
        print(f"    This will cause wrong node features to be used!")
    else:
        print(f"  ✓ Edge indices within bounds")
    
    # Issue 2: Edge indices shifted by 1 (common bug)
    zero_edges = (edge_index == 0).sum()
    max_edges = (edge_index == sample.x.shape[0] - 1).sum()
    print(f"  • Edges using node 0: {zero_edges}")
    print(f"  • Edges using node {sample.x.shape[0]-1}: {max_edges}")
    
    # Issue 3: Check strain value distribution
    strains = data['strains'][0].numpy().flatten()
    non_zero = strains[strains != 0]
    print(f"\n  • Strain values:")
    print(f"    Non-zero strains: {len(non_zero)} / {len(strains)}")
    print(f"    Range: [{non_zero.min():.6f}, {non_zero.max():.6f}]")
    print(f"    Mean: {non_zero.mean():.6f}, Std: {non_zero.std():.6f}")
    
    if non_zero.std() < 1e-6:
        print(f"  ⚠ WARNING: Very low strain variance - check if data is correct!")


def validate_data_consistency(data):
    """Check data consistency and report any issues."""
    print(f"\n{'='*60}")
    print("DATA CONSISTENCY CHECKS")
    print(f"{'='*60}\n")
    
    issues = []
    
    # Check if all datasets have same number of samples
    num_samples_gnn = len(data.get('node_gnn_data', []))
    num_samples_strains = data['strains'].shape[0] if 'strains' in data else 0
    num_samples_geom = len(data.get('geometry', [])) if 'geometry' in data else 0
    
    print(f"Sample counts:")
    print(f"  - Node GNN data: {num_samples_gnn}")
    print(f"  - Strains: {num_samples_strains}")
    if 'geometry' in data:
        print(f"  - Geometry: {num_samples_geom}")
    
    if num_samples_gnn != num_samples_strains:
        issues.append(f"⚠ Mismatch: GNN data has {num_samples_gnn} samples but strains has {num_samples_strains}")
    
    # Check for NaN or Inf values
    if 'strains' in data:
        has_nan = torch.isnan(data['strains']).any()
        has_inf = torch.isinf(data['strains']).any()
        if has_nan:
            issues.append("⚠ Strains contain NaN values")
        if has_inf:
            issues.append("⚠ Strains contain Inf values")
    
    # Check node feature dimensions
    if 'node_gnn_data' in data and len(data['node_gnn_data']) > 0:
        feature_dims = [d.x.shape[0] for d in data['node_gnn_data']]
        print(f"\nNode counts per sample (first 10): {feature_dims[:10]}")
        
        # Check strain tensor dimensions match node counts
        if 'strains' in data:
            strain_sample_sizes = []
            for i in range(min(10, len(data['node_gnn_data']))):
                num_nodes = data['node_gnn_data'][i].x.shape[0]
                num_strains = (data['strains'][i] != 0).sum().item()
                strain_sample_sizes.append((num_nodes, num_strains))
            
            print(f"Node vs non-zero strain counts (first 10):")
            for i, (n_nodes, n_strains) in enumerate(strain_sample_sizes):
                match_symbol = "✓" if n_nodes == n_strains else "~"
                print(f"  Sample {i}: {n_nodes} nodes vs {n_strains} strains {match_symbol}")
            
            # Allow small mismatches (some boundary nodes may not have strain data)
            large_mismatches = sum(1 for n, s in strain_sample_sizes if abs(n - s) > n * 0.05)  # >5% missing
            if large_mismatches > 0:
                issues.append(f"⚠ {large_mismatches}/10 samples have significant node-strain count mismatch (>5%)")
            else:
                print(f"  ℹ Minor strain data gaps detected (expected for boundary nodes)")
    
    print()
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ All consistency checks passed!")
    
    print(f"\n{'='*60}\n")
    
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Validate and visualize processed FCLGA data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help='Starting sample index to visualize (default: 0)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Number of samples to visualize (default: 3)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots (only check data)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("FCLGA GraphTransformer - Data Validation")
    print(f"{'='*60}")
    
    # Load data
    data = load_data()
    
    if not data:
        print("\n✗ No data found! Run preprocessing first.")
        return 1
    
    # Validate consistency
    is_consistent = validate_data_consistency(data)
    
    # Compare with legacy format
    compare_with_legacy_dataset(data)
    
    if args.no_plots:
        print("Skipping plots (--no-plots flag set)")
        return 0 if is_consistent else 1
    
    # Generate visualizations
    print(f"{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")
    
    num_samples = min(args.num_samples, len(data.get('node_gnn_data', [])))
    all_issues = []
    
    for i in range(args.sample, args.sample + num_samples):
        if i >= len(data['node_gnn_data']):
            break
        
        print(f"Processing sample {i}...")
        
        # Plot mesh geometry
        mesh_path = RESULTS_PLOTS / f"sample_{i:03d}_mesh.png"
        plot_mesh_geometry(data, i, mesh_path)
        
        # Plot edge connectivity (NEW - CRITICAL FOR DEBUGGING)
        edge_path = RESULTS_PLOTS / f"sample_{i:03d}_edge_connectivity.png"
        issues = plot_edge_connectivity(data, i, edge_path)
        if issues:
            all_issues.extend(issues)
        
        # Plot strain field
        strain_path = RESULTS_PLOTS / f"sample_{i:03d}_strain_field.png"
        plot_strain_field(data, i, strain_path)
        
        print()
    
    # Plot overall statistics
    print("Generating dataset statistics...")
    stats_path = RESULTS_PLOTS / "dataset_statistics.png"
    plot_statistics(data, stats_path)
    
    print(f"\n{'='*60}")
    print("✓ VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nPlots saved to: {RESULTS_PLOTS}")
    print(f"Generated {num_samples * 3 + 1} visualization files")
    
    if all_issues:
        print(f"\n⚠ ISSUES DETECTED:")
        for issue in set(all_issues):
            print(f"  {issue}")
    
    print(f"\nData quality: {'✓ PASS' if is_consistent and not all_issues else '✗ ISSUES FOUND'}")
    
    return 0 if (is_consistent and not all_issues) else 1


if __name__ == '__main__':
    sys.exit(main())
