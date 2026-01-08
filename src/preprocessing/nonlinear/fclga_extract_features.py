"""Extract node features and connectivity from Abaqus .inp files.

Parses Abaqus input files to extract mesh geometry and converts them
into PyTorch Geometric graph structures for GNN training.
Also generates triangulation data for visualization.

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
GEOMETRY_DIR = PROJECT_ROOT / "data" / "raw" / "nonlinear" / "geometry"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "nonlinear"

# Create output directory
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def parse_inp_file_for_nodes(filepath):
    """Parse Abaqus .inp file to extract nodes and elements.

    Reads node coordinates and element connectivity from an Abaqus input file.
    The parser handles multi-section files by tracking section headers.

    """
    data = {"nodes": [], "elements": []}

    with open(filepath) as f:
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
                parts = line.split(",")
                element_id = int(parts[0].strip())
                node_indices = [int(node.strip()) for node in parts[1:] if node.strip()]
                data["elements"].append((element_id, node_indices))
            continue

    return data


def build_node_edge_index(elements):
    """Build bidirectional edge index from element connectivity.

    Converts element connectivity (faces/cells) into a graph edge list
    by connecting all nodes within each element.

    """
    edge_set = set()

    for _, nodes in elements:
        num_nodes = len(nodes)
        # Connect all nodes within each element
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_set.add((nodes[i] - 1, nodes[j] - 1))
                edge_set.add((nodes[j] - 1, nodes[i] - 1))

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    return edge_index


def save_triangulation_data(all_data, directory):
    """Save triangulation data for visualization purposes.

    Creates matplotlib Triangulation objects from mesh data for contour plotting.

    Args:
        all_data: List of tuples (parsed_data, data_obj) for each sample
        directory: Output directory path
    """
    triangulation_data = {}

    for i, (data, data_obj) in enumerate(all_data):
        node_positions = data_obj.x.numpy()
        # Convert to 0-indexed
        elements = [[node_id - 1 for node_id in element[1]] for element in data["elements"]]

        nodes_x = node_positions[:, 0]
        nodes_y = node_positions[:, 1]

        triangulation = tri.Triangulation(nodes_x, nodes_y, elements)
        triangulation_data[f"sample_{i}"] = triangulation

    output_path = os.path.join(directory, "triangulation_data.pkl")
    with open(output_path, "wb") as file:
        pickle.dump(triangulation_data, file)

    print(f"✓ Triangulation data saved to: {output_path}")


def pad_features(features_list, max_len):
    """Pad feature arrays to uniform length for batching.

    Args:
        features_list: List of numpy arrays with shape [n_nodes, n_features]
        max_len: Target length for padding

    Returns:
        np.ndarray: Padded features array [n_samples, max_len, n_features]
    """
    padded_features = []
    for features in features_list:
        num_elements = features.shape[0]

        if num_elements < max_len:
            padding = np.zeros((max_len - num_elements, features.shape[1]))
            padded_features.append(np.vstack((features, padding)))
        else:
            padded_features.append(features)
    return np.array(padded_features)


def prepare_node_data_for_gnn(directory=None):
    """Parse all .inp files and create PyTorch Geometric graph data.

    Main processing function that reads all Abaqus input files, extracts
    node positions and connectivity, and creates graph structures suitable
    for GNN training.

    """
    if directory is None:
        directory = GEOMETRY_DIR
    else:
        directory = Path(directory)

    # Get all .inp files sorted by sample number
    # Regex matches number in filename: Plate_nonlinear_0.inp -> 0
    inp_files = sorted(
        directory.glob("*.inp"),
        key=lambda x: int(re.search(r"(\d+)", x.stem).group())
        if re.search(r"(\d+)", x.stem)
        else 0,
    )

    if not inp_files:
        raise FileNotFoundError(f"No .inp files found in {directory}")

    print(f"\nProcessing {len(inp_files)} geometry files...")

    data_list = []
    parsed_data_list = []
    max_node_count = 0

    for inp_file in inp_files:
        filepath = str(inp_file)
        print(f"  Processing: {inp_file.name}")
        data = parse_inp_file_for_nodes(filepath)
        edge_index = build_node_edge_index(data["elements"])

        nodes = sorted(data["nodes"], key=lambda x: x[0])
        node_features = torch.tensor([[x, y] for _, x, y in nodes], dtype=torch.float)
        node_ids = torch.tensor([node_id for node_id, _, _ in nodes], dtype=torch.long)

        data_obj = Data(x=node_features, edge_index=edge_index, node_ids=node_ids)
        data_list.append(data_obj)
        parsed_data_list.append(data)
        max_node_count = max(max_node_count, len(nodes))
        print(f"    Nodes: {len(nodes)}, Node ID range: [{node_ids.min()}, {node_ids.max()}]")

    # Pad node features for all samples
    padded_node_features = pad_features([data.x.numpy() for data in data_list], max_node_count)

    # Update data objects with padded features
    for i, data_obj in enumerate(data_list):
        data_obj.x = torch.tensor(padded_node_features[i], dtype=torch.float)

    # Save graph data
    node_data_path = DATA_PROCESSED / "node_gnn_data.pt"
    torch.save(data_list, str(node_data_path))
    print(f"✓ Node data saved to: {node_data_path}")

    # Save triangulation data for visualization
    save_triangulation_data(list(zip(parsed_data_list, data_list)), DATA_PROCESSED)


def plot_sample(data_obj, data, output_dir=None):
    """Plot mesh geometry and triangulation for visual inspection.

    Creates two plots: node connectivity graph and triangulation mesh.
    Useful for debugging and validation.

    Args:
        data_obj: PyTorch Geometric Data object with graph structure
        data: Parsed data dictionary with elements
        output_dir: Directory to save plots (default: current directory)
    """
    node_positions = data_obj.x.numpy()
    edge_index = data_obj.edge_index.numpy()

    # Plot node connectivity
    plt.figure(figsize=(10, 10))
    plt.scatter(node_positions[:, 0], node_positions[:, 1], c="blue", label="Nodes")

    for edge in edge_index.T:
        node_start, node_end = edge
        x_coords = [node_positions[node_start, 0], node_positions[node_end, 0]]
        y_coords = [node_positions[node_start, 1], node_positions[node_end, 1]]
        plt.plot(x_coords, y_coords, "r-", alpha=0.5)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Node Connectivity")
    plt.legend()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "node_connectivity.png"))
    else:
        plt.savefig("node_connectivity.png")
    plt.close()

    # Plot triangulation
    elements = [[node_id - 1 for node_id in element[1]] for element in data["elements"]]
    nodes_x = node_positions[:, 0]
    nodes_y = node_positions[:, 1]
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements)

    plt.figure(figsize=(10, 10))
    plt.triplot(triangulation, "go-", alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Triangulation")

    if output_dir:
        plt.savefig(os.path.join(output_dir, "triangulation.png"))
    else:
        plt.savefig("triangulation.png")
    plt.close()


if __name__ == "__main__":
    print(f"Reading geometry files from: {GEOMETRY_DIR}")
    print(f"Output will be saved to: {DATA_PROCESSED}")

    # Run the parser and prepare node data
    prepare_node_data_for_gnn()
