"""Extract node features and connectivity from elastic case Abaqus .inp files.

Parses Abaqus input files to extract mesh geometry and converts them
into PyTorch Geometric graph structures for GNN training.
Also generates triangulation data for visualization.

This version maintains legacy elastic compatibility with +1 padding.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import torch
from torch_geometric.data import Data


# Setup organized paths
PROJECT_ROOT = Path.cwd()
GEOMETRY_DIR = PROJECT_ROOT / "data" / "raw" / "linear" / "geometry"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "linear"

# Create output directory
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def parse_inp_file_for_nodes(filepath):
    """Parse Abaqus .inp file to extract nodes and elements.
    
    Reads node coordinates and element connectivity from an Abaqus input file.
    The parser handles multi-section files by tracking section headers.
    
    Args:
        filepath: Path to .inp file
    
    Returns:
        dict: Contains 'nodes' list of (id, x, y) and 'elements' list of (id, [node_ids])
    """
    data = {
        "nodes": [],
        "elements": []
    }

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_node_section = False
    in_element_section = False

    for line in lines:
        line = line.strip()

        # Start of node section
        if line.startswith("*Node"):
            if line.startswith("*Node Output"):
                continue
            else:
                in_node_section = True
                continue

        # Start of element section
        if line.startswith("*Element,"):
            in_node_section = False
            in_element_section = True
            continue

        # Parse node data
        if in_node_section:
            node_data = line.split(",")
            node_id = int(node_data[0].strip())
            x_pos = float(node_data[1].strip())
            y_pos = float(node_data[2].strip())
            data["nodes"].append((node_id, x_pos, y_pos))
            continue

        # Parse element data
        if in_element_section:
            if line.startswith("*"):
                in_element_section = False
            elif line:
                parts = line.split(',')
                element_id = int(parts[0].strip())
                node_indices = [int(node.strip()) for node in parts[1:] if node.strip()]
                data["elements"].append((element_id, node_indices))
            continue

    return data


def build_node_edge_index(elements):
    """Build bidirectional edge index from element connectivity.
    
    Creates graph edges by connecting all nodes within each element.
    Automatically creates bidirectional edges for message passing.
    
    Args:
        elements: List of (element_id, [node_ids])
    
    Returns:
        torch.Tensor: Edge index [2, num_edges] with bidirectional connections
    """
    edge_set = set()

    for _, nodes in elements:
        num_nodes = len(nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Add both directions
                edge_set.add((nodes[i] - 1, nodes[j] - 1))
                edge_set.add((nodes[j] - 1, nodes[i] - 1))

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    return edge_index


def save_triangulation_data(all_data, directory="."):
    """Save triangulation data for visualization.
    
    Creates matplotlib triangulation objects for each sample and saves
    them to a pickle file for later use in result visualization.
    
    Args:
        all_data: List of (parsed_data_dict, Data_object) tuples
        directory: Output directory path
    """
    triangulation_data = {}
    
    for i, (data, data_obj) in enumerate(all_data):
        node_positions = data_obj.x.numpy()
        elements = [[node_id - 1 for node_id in element[1]] for element in data["elements"]]

        nodes_x = node_positions[:, 0]
        nodes_y = node_positions[:, 1]

        triangulation = tri.Triangulation(nodes_x, nodes_y, elements)
        triangulation_data[f'sample_{i}'] = triangulation

    output_path = os.path.join(directory, 'triangulation_data.pkl')
    with open(output_path, 'wb') as file:
        pickle.dump(triangulation_data, file)

    print(f"✓ Triangulation data saved to: {output_path}")


def pad_features(features_list, max_len):
    """Pad feature arrays to uniform length with zeros.
    
    Args:
        features_list: List of numpy arrays with shape [num_nodes, feature_dim]
        max_len: Target length for padding
    
    Returns:
        numpy.ndarray: Padded features [num_samples, max_len, feature_dim]
    """
    padded_features = []
    for features in features_list:
        num_elements = features.shape[0]

        if num_elements < max_len:
            # Pad with zeros to match max_len
            padding = np.zeros((max_len - num_elements, features.shape[1]))
            padded_features.append(np.vstack((features, padding)))
        else:
            padded_features.append(features)
    
    return np.array(padded_features)


def prepare_node_data_for_gnn(directory="."):
    """Extract and prepare node data from all .inp files.
    
    Main processing function that:
    1. Finds all .inp files in directory
    2. Extracts nodes and elements from each file
    3. Creates PyTorch Geometric Data objects
    4. Pads features to uniform size
    5. Saves node data and triangulation data
    
    Args:
        directory: Directory containing .inp files
    """
    print("=" * 80)
    print("ELASTIC CASE: EXTRACTING FEATURES FROM .INP FILES")
    print("=" * 80)
    
    # Find all .inp files
    inp_files = [f for f in os.listdir(directory) if f.endswith(".inp")]
    
    # Sort numerically by extracting sample number from filename
    inp_files.sort(key=lambda x: int(re.search(r'(\d+)', x.split('_')[-1]).group()))
    
    print(f"Found {len(inp_files)} .inp files")
    print("-" * 80)

    data_list = []
    parsed_data_list = []
    max_node_count = 0

    for inp_file in inp_files:
        filepath = os.path.join(directory, inp_file)
        data = parse_inp_file_for_nodes(filepath)
        edge_index = build_node_edge_index(data["elements"])

        nodes = sorted(data["nodes"], key=lambda x: x[0])
        node_features = torch.tensor([[x, y] for _, x, y in nodes], dtype=torch.float)

        data_obj = Data(x=node_features, edge_index=edge_index)
        data_list.append(data_obj)
        parsed_data_list.append(data)
        max_node_count = max(max_node_count, len(nodes))
        
        if len(data_list) % 50 == 0 or len(data_list) == 1:
            print(f"Processed {len(data_list)}/{len(inp_files)}: {inp_file} ({len(nodes)} nodes)")

    print("-" * 80)
    print(f"Maximum node count: {max_node_count}")
    
    # LEGACY ELASTIC: Pad with +1 for compatibility
    print(f"Padding to: {max_node_count + 1} (max_node_count + 1)")
    padded_node_features = pad_features([data.x.numpy() for data in data_list], max_node_count + 1)

    # Update data_list with padded features
    for i, data_obj in enumerate(data_list):
        data_obj.x = torch.tensor(padded_node_features[i], dtype=torch.float)

    # Save node graph data
    output_file = os.path.join(directory, "node_gnn_data.pt")
    torch.save(data_list, output_file)
    print(f"✓ Node data saved to: {output_file}")

    # Save triangulation data for visualization
    save_triangulation_data(list(zip(parsed_data_list, data_list)), directory)
    
    print("=" * 80)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 80)


def plot_sample(data_obj, data, output_dir="."):
    """Plot node connectivity and triangulation for debugging.
    
    Args:
        data_obj: PyTorch Geometric Data object
        data: Parsed data dictionary
        output_dir: Directory to save plots
    """
    node_positions = data_obj.x.numpy()
    edge_index = data_obj.edge_index.numpy()

    # Plot 1: Node connectivity graph
    plt.figure(figsize=(10, 10))
    plt.scatter(node_positions[:, 0], node_positions[:, 1], c='blue', label='Nodes', s=20)

    for edge in edge_index.T:
        node_start, node_end = edge
        x_coords = [node_positions[node_start, 0], node_positions[node_end, 0]]
        y_coords = [node_positions[node_start, 1], node_positions[node_end, 1]]
        plt.plot(x_coords, y_coords, 'r-', alpha=0.3, linewidth=0.5)

    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Node Connectivity Graph')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'node_connectivity.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Triangulation mesh
    elements = [[node_id - 1 for node_id in element[1]] for element in data["elements"]]
    nodes_x = node_positions[:, 0]
    nodes_y = node_positions[:, 1]
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements)

    plt.figure(figsize=(10, 10))
    plt.triplot(triangulation, 'go-', alpha=0.5, linewidth=0.8)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Triangulation Mesh')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'triangulation_mesh.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plots saved to: {output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Run the parser and prepare node data from elastic geometry directory
    prepare_node_data_for_gnn(directory=str(GEOMETRY_DIR))
