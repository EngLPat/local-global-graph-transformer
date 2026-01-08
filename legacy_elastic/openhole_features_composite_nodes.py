import os
import re
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
import pickle
import matplotlib.tri as tri
import matplotlib.pyplot as plt

def parse_inp_file_for_nodes(filepath):
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

        if line.startswith("*Node"):
            if line.startswith("*Node Output"):
                continue
            else:
                in_node_section = True
                continue

        if line.startswith("*Element,"):
            in_node_section = False
            in_element_section = True
            continue

        if in_node_section:
            node_data = line.split(",")
            node_id = int(node_data[0].strip())
            x_pos = float(node_data[1].strip())
            y_pos = float(node_data[2].strip())
            data["nodes"].append((node_id, x_pos, y_pos))
            continue

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
    edge_set = set()

    for _, nodes in elements:
        num_nodes = len(nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_set.add((nodes[i] - 1, nodes[j] - 1))
                edge_set.add((nodes[j] - 1, nodes[i] - 1))

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    return edge_index

def save_triangulation_data(all_data, directory="."):
    triangulation_data = {}
    
    for i, (data, data_obj) in enumerate(all_data):
        node_positions = data_obj.x.numpy()
        elements = [[node_id - 1 for node_id in element[1]] for element in data["elements"]]

        nodes_x = node_positions[:, 0]
        nodes_y = node_positions[:, 1]

        triangulation = tri.Triangulation(nodes_x, nodes_y, elements)
        triangulation_data[f'sample_{i}'] = triangulation

    with open(os.path.join(directory, 'triangulation_data.pkl'), 'wb') as file:
        pickle.dump(triangulation_data, file)

    print("Triangulation data for all samples saved.")

def pad_features(features_list, max_len):
    """Pad each feature vector or target array to a uniform length."""
    padded_features = []
    for features in features_list:
        num_elements = features.shape[0]

        if num_elements < max_len:
            # Pad with zeros (or another placeholder) to the required shape
            padding = np.zeros((max_len - num_elements, features.shape[1]))
            padded_features.append(np.vstack((features, padding)))
        else:
            padded_features.append(features)
    return np.array(padded_features)

def prepare_node_data_for_gnn(directory="."):
    inp_files = [f for f in os.listdir(directory) if f.endswith(".inp")]
    inp_files.sort(key=lambda x: int(re.search(r'(\d+)', x.split('_')[-1]).group()))

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
        print(f"Processed {inp_file}")

    # Pad node features for all samples
    padded_node_features = pad_features([data.x.numpy() for data in data_list], max_node_count+1)

    # Update data_list with padded features
    for i, data_obj in enumerate(data_list):
        data_obj.x = torch.tensor(padded_node_features[i], dtype=torch.float)

    torch.save(data_list, "node_gnn_data.pt")
    print("Node data saved for GNN input.")

    # Save triangulation data
    save_triangulation_data(list(zip(parsed_data_list, data_list)), directory)

    # Plot the first sample
    # plot_sample(data_list[0], parsed_data_list[0])

def plot_sample(data_obj, data):
    node_positions = data_obj.x.numpy()
    edge_index = data_obj.edge_index.numpy()

    plt.figure(figsize=(10, 10))
    plt.scatter(node_positions[:, 0], node_positions[:, 1], c='blue', label='Nodes')

    for edge in edge_index.T:
        node_start, node_end = edge
        x_coords = [node_positions[node_start, 0], node_positions[node_end, 0]]
        y_coords = [node_positions[node_start, 1], node_positions[node_end, 1]]
        plt.plot(x_coords, y_coords, 'r-', alpha=0.5)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Node Connectivity')
    plt.legend()
    plt.savefig()

    # Plot triangulation
    elements = [[node_id - 1 for node_id in element[1]] for element in data["elements"]]
    nodes_x = node_positions[:, 0]
    nodes_y = node_positions[:, 1]
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements)

    plt.figure(figsize=(10, 10))
    plt.triplot(triangulation, 'go-', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Triangulation')
    plt.savefig()

# Run the parser and prepare node data
prepare_node_data_for_gnn()