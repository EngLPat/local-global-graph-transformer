"""
Prepare Training Data for FCLGA GraphTransformer - ELASTIC CASE.

This script combines preprocessed data (node positions, triangulation, strains, geometry)
into complete graph data objects ready for training. It adds:
- Node features: positions, boundary conditions, displacement amounts
- Edge features: relative positions and distances
- Target strain values

ELASTIC CASE: Fabric composite material with Static/Implicit solver.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import argparse
import pickle

# Import configuration
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import paths as config_paths


def load_preprocessed_data():
    """Load all preprocessed data files for elastic case."""
    print("Loading preprocessed data...")

    # Get linear-specific paths
    paths_linear = config_paths.get_paths("linear")

    gnn_data = torch.load(paths_linear.NODE_DATA, weights_only=False)
    strains = torch.load(paths_linear.STRAINS_TENSOR, weights_only=False)

    # Try both .pt and .pkl for geometry data
    if paths_linear.GEOMETRY_TENSOR.exists():
        plate_geometry_data = torch.load(paths_linear.GEOMETRY_TENSOR, weights_only=False)
    elif paths_linear.GEOMETRY_PICKLE.exists():
        with open(paths_linear.GEOMETRY_PICKLE, "rb") as f:
            plate_geometry_data = pickle.load(f)
            # Convert to tensor if it's a list
            if isinstance(plate_geometry_data, list):
                plate_geometry_data = torch.tensor(plate_geometry_data, dtype=torch.float)
    else:
        raise FileNotFoundError("Geometry data not found in .pt or .pkl format")

    with open(paths_linear.TRIANGULATION_DATA, "rb") as f:
        triangulation_data = pickle.load(f)

    print(f"✓ Loaded {len(gnn_data)} graph samples")
    print(f"✓ Loaded {len(strains)} strain samples")
    print(f"✓ Loaded geometry data: {plate_geometry_data.shape}")

    return gnn_data, strains, plate_geometry_data, triangulation_data


def detect_hole_edge_nodes(node_positions, hole_x, hole_y, hole_radius, tolerance_factor=0.05):
    """
    Detect nodes on the hole edge using distance criterion.

    Parameters
    ----------
    node_positions : torch.Tensor
        Node coordinates [N, 2]
    hole_x : float
        Hole center x-coordinate
    hole_y : float
        Hole center y-coordinate
    hole_radius : float
        Hole radius
    tolerance_factor : float, optional
        Tolerance as fraction of radius (default: 0.05 = 5%)

    Returns
    -------
    torch.Tensor
        Binary indicator [N, 1] for hole edge nodes
    torch.Tensor
        Distances from hole center
    float
        Tolerance value used
    """
    delta_x = node_positions[:, 0] - hole_x
    delta_y = node_positions[:, 1] - hole_y
    distances_from_hole = torch.sqrt(delta_x**2 + delta_y**2)

    tolerance = tolerance_factor * hole_radius
    is_hole_edge = torch.abs(distances_from_hole - hole_radius) < tolerance

    return is_hole_edge.float().unsqueeze(1), distances_from_hole, tolerance


def create_node_features(
    node_positions, hole_x, hole_y, hole_radius, displacement, fixed_threshold=0.7
):
    """
    Create complete node feature vectors.

    Features: [x, y, is_hole_edge, is_fixed, is_displaced, displacement_amount]

    Parameters
    ----------
    node_positions : torch.Tensor
        Node coordinates [N, 2]
    hole_x : float
        Hole center x-coordinate
    hole_y : float
        Hole center y-coordinate
    hole_radius : float
        Hole radius
    displacement : float
        Applied displacement value
    fixed_threshold : float, optional
        X-coordinate threshold for fixed boundary (default: 0.7)

    Returns
    -------
    torch.Tensor
        Complete node features [N, 6]
    dict
        Statistics about detected features
    """
    # Detect hole edge nodes
    is_hole_edge, distances_from_hole, tolerance = detect_hole_edge_nodes(
        node_positions, hole_x, hole_y, hole_radius
    )

    # Detect boundary conditions
    fixed_nodes = node_positions[:, 0] <= fixed_threshold
    loaded_nodes = node_positions[:, 0] >= (node_positions[:, 0].max() - 1e-1)

    # Create feature tensors
    is_fixed = torch.zeros(node_positions.shape[0], 1)
    is_fixed[fixed_nodes] = 1.0

    is_displaced = torch.zeros(node_positions.shape[0], 1)
    is_displaced[loaded_nodes] = 1.0

    displacement_amount = torch.zeros(node_positions.shape[0], 1)
    displacement_amount[loaded_nodes] = displacement

    # Concatenate all features
    x = torch.cat(
        (node_positions, is_hole_edge, is_fixed, is_displaced, displacement_amount), dim=1
    )

    stats = {
        "hole_edge_nodes": is_hole_edge.sum().item(),
        "fixed_nodes": fixed_nodes.sum().item(),
        "displaced_nodes": loaded_nodes.sum().item(),
        "distances_min": distances_from_hole.min().item(),
        "distances_max": distances_from_hole.max().item(),
        "tolerance": tolerance,
    }

    return x, stats


def create_edge_features(edge_index, node_positions):
    """
    Create edge feature vectors with relative positions and distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge connectivity [2, E]
    node_positions : torch.Tensor
        Node coordinates [N, 2]

    Returns
    -------
    torch.Tensor
        Edge features [E, 3] = [dx, dy, distance]
    """
    u_i = node_positions[edge_index[0]]  # Start node positions
    u_j = node_positions[edge_index[1]]  # End node positions
    u_ij = u_i - u_j  # Relative positions
    u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)  # Distances

    edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).type(torch.float)
    return edge_attr


def make_graph_bidirectional(edge_index, node_positions):
    """
    Ensure graph has bidirectional edges and compute edge attributes.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge connectivity [2, E]
    node_positions : torch.Tensor
        Node coordinates [N, 2]

    Returns
    -------
    tuple
        (edge_index, edge_attr) with bidirectional edges
    """
    num_edges = edge_index.shape[1]
    edge_index_set = set(
        (edge_index[0, i].item(), edge_index[1, i].item()) for i in range(num_edges)
    )

    # Check if reverse edges are needed
    needs_reverse_edges = False
    for j in range(num_edges):
        src, dst = edge_index[0, j].item(), edge_index[1, j].item()
        if (dst, src) not in edge_index_set:
            needs_reverse_edges = True
            break

    if needs_reverse_edges:
        # Add reverse edges
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)

        # Compute edge attributes for original edges
        u_i = node_positions[edge_index[0, :num_edges]]
        u_j = node_positions[edge_index[1, :num_edges]]
        u_ij = u_i - u_j
        u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
        edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).type(torch.float)

        # For reverse edges, negate relative positions (norm stays same)
        u_ij_reverse = -u_ij
        edge_attr_reverse = torch.cat((u_ij_reverse, u_ij_norm), dim=-1).type(torch.float)

        edge_attr = torch.cat([edge_attr, edge_attr_reverse], dim=0)
    else:
        edge_attr = create_edge_features(edge_index, node_positions)

    return edge_index, edge_attr


def process_dataset(gnn_data, strains, plate_geometry_data, triangulation_data):
    """
    Process all samples to create complete graph data objects.

    Parameters
    ----------
    gnn_data : list
        List of graph data with positions and connectivity
    strains : list
        List of strain tensors (target values)
    plate_geometry_data : torch.Tensor
        Geometry parameters [N, 6]
    triangulation_data : dict
        Dictionary of triangulation objects

    Returns
    -------
    list
        List of torch_geometric.data.Data objects ready for training
    """
    print(f"\nProcessing {len(gnn_data)} samples...")

    processed_data_list = []

    # Check if first sample needs bidirectional edges
    sample_edge_index = gnn_data[0].edge_index
    num_edges = sample_edge_index.shape[1]
    edge_set = set(
        (sample_edge_index[0, j].item(), sample_edge_index[1, j].item()) for j in range(num_edges)
    )

    needs_reverse_edges = any(
        (dst, src) not in edge_set
        for src, dst in [
            (sample_edge_index[0, j].item(), sample_edge_index[1, j].item())
            for j in range(num_edges)
        ]
    )

    if needs_reverse_edges:
        print("Graph is NOT bidirectional. Adding reverse edges for all samples.")
    else:
        print("Graph is already bidirectional.")

    # Process each sample
    for i, (strain, data) in enumerate(zip(strains, gnn_data)):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing sample {i + 1}/{len(gnn_data)}...")

        # Extract triangulation data
        triangulation = triangulation_data[f"sample_{i}"]
        cells = torch.tensor(triangulation.triangles, dtype=torch.long)
        mesh_pos = torch.tensor(
            np.column_stack((triangulation.x, triangulation.y)), dtype=torch.float
        )

        # Get node positions and connectivity
        node_positions = data.x
        edge_index = data.edge_index

        # Extract geometry parameters
        L = plate_geometry_data[i][0].item()
        W = plate_geometry_data[i][1].item()
        hole_radius = plate_geometry_data[i][2].item()
        hole_x = plate_geometry_data[i][3].item()
        hole_y = plate_geometry_data[i][4].item()
        displacement = plate_geometry_data[i][5].item()

        # Create complete node features
        x, stats = create_node_features(node_positions, hole_x, hole_y, hole_radius, displacement)

        if i == 0:
            # Print statistics for first sample
            print("\nFirst sample statistics:")
            print(
                f"  Geometry: L={L:.2f}, W={W:.2f}, radius={hole_radius:.2f}, displacement={displacement:.4f}"
            )
            print(f"  Hole edge nodes: {stats['hole_edge_nodes']}")
            print(f"  Fixed boundary nodes: {stats['fixed_nodes']}")
            print(f"  Displaced boundary nodes: {stats['displaced_nodes']}")
            print(f"  Distance tolerance: {stats['tolerance']:.4f}")

        # Create bidirectional edges and edge features
        edge_index, edge_attr = make_graph_bidirectional(edge_index, node_positions)

        # Create complete Data object
        data_obj = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=strain,
            cells=cells,
            mesh_pos=mesh_pos,
        )

        processed_data_list.append(data_obj)

    print(f"✓ Processed all {len(processed_data_list)} samples")
    return processed_data_list


def visualize_features(data_list, output_path):
    """
    Create visualization of node features for first sample.

    Parameters
    ----------
    data_list : list
        List of Data objects
    output_path : Path
        Path to save visualization
    """
    print("\nGenerating feature visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    sample_idx = 0
    sample = data_list[sample_idx]
    node_positions = sample.x[:, :2].numpy()

    features = [
        "Node Positions",
        "Is Hole Edge",
        "Is Fixed",
        "Is Displaced",
        "Displacement Amount",
        "Strain (Target)",
    ]

    # Node positions plot
    axes[0].scatter(node_positions[:, 0], node_positions[:, 1], c="blue", s=10, alpha=0.6)
    axes[0].set_title(features[0])
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].axis("equal")

    # Plot each feature
    for idx, (feature_idx, feature_name) in enumerate(
        [(2, features[1]), (3, features[2]), (4, features[3]), (5, features[4])], start=1
    ):
        sc = axes[idx].scatter(
            node_positions[:, 0],
            node_positions[:, 1],
            c=sample.x[:, feature_idx].numpy(),
            cmap="RdYlGn" if idx == 1 else "viridis",
            s=10,
            alpha=0.6,
        )
        axes[idx].set_title(feature_name)
        axes[idx].set_xlabel("X")
        axes[idx].set_ylabel("Y")
        axes[idx].axis("equal")
        fig.colorbar(sc, ax=axes[idx])

    # Plot strain (output) - assuming first component
    strain_component = sample.y[:, 0].numpy() if sample.y.dim() > 1 else sample.y.numpy()
    sc = axes[5].scatter(
        node_positions[:, 0],
        node_positions[:, 1],
        c=strain_component,
        cmap="plasma",
        s=10,
        alpha=0.6,
    )
    axes[5].set_title(features[5])
    axes[5].set_xlabel("X")
    axes[5].set_ylabel("Y")
    axes[5].axis("equal")
    fig.colorbar(sc, ax=axes[5])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Prepare complete training dataset with node and edge features - ELASTIC CASE"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate feature visualization plot"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ELASTIC CASE: FCLGA Training Data Preparation")
    print("=" * 70)

    # Load preprocessed data
    gnn_data, strains, plate_geometry_data, triangulation_data = load_preprocessed_data()

    # Verify data consistency
    assert len(gnn_data) == len(strains), "Mismatch in number of samples"
    assert len(gnn_data) == len(plate_geometry_data), "Mismatch in geometry data"

    # Process dataset
    processed_data_list = process_dataset(
        gnn_data, strains, plate_geometry_data, triangulation_data
    )

    # Save processed data
    paths_linear = config_paths.get_paths("linear")
    output_path = paths_linear.DATASETS_DIR / "processed_data.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(processed_data_list, output_path)
    print(f"\n✓ Saved processed data to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Data Summary")
    print("=" * 70)
    print(f"Number of samples: {len(processed_data_list)}")
    print("Node features: [x, y, is_hole_edge, is_fixed, is_displaced, displacement_amount]")
    print(f"Node feature shape: {processed_data_list[0].x.shape}")
    print(f"Edge index shape: {processed_data_list[0].edge_index.shape}")
    print(f"Edge attribute shape: {processed_data_list[0].edge_attr.shape}")
    print(f"Target shape: {processed_data_list[0].y.shape}")
    print("=" * 70)

    # Optional visualization
    if args.visualize:
        viz_path = paths_linear.RESULTS_DIR / "preprocessing" / "feature_visualization.png"
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_features(processed_data_list, viz_path)

    print("\n✓ Training data preparation complete!")


if __name__ == "__main__":
    main()
