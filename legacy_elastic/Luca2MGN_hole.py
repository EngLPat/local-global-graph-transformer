import torch
import pickle
import os
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt

# Define file paths
gnn_data_path = os.path.join(".",  'node_gnn_data.pt')
triangulation_data_path = os.path.join(".", 'triangulation_data.pkl')
strains_path = os.path.join(".", 'strains.pt')
plate_geometry_data_path = os.path.join(".", 'plate_geometry_data.pt')

# Load datasets
gnn_data = torch.load(gnn_data_path, weights_only=False)
strains = torch.load(strains_path, weights_only=False)
plate_geometry_data = torch.load(plate_geometry_data_path, weights_only=False)

with open(triangulation_data_path, 'rb') as f:
    triangulation_data = pickle.load(f)

# Print data shape to understand the structure
print(f"Plate geometry data shape: {plate_geometry_data.shape}")
print(f"First geometry sample: {plate_geometry_data[0]}")

# Create missing data
data_list = []
print(f"Processing {len(strains)} samples...")

for i, (strain, data) in enumerate(zip(strains, gnn_data)):
    # Only print progress for every 10th sample or the last one
    if i % 10 == 0 or i == len(strains) - 1:
        print(f"Processing sample {i+1}/{len(strains)}")
        
    x = data.x
    edge_index = data.edge_index
    y = strain

    # Extract cells and mesh_pos from triangulation data
    triangulation = triangulation_data[f'sample_{i}']
    cells = torch.tensor(triangulation.triangles).type(torch.long)
    mesh_pos = torch.tensor(np.column_stack((triangulation.x, triangulation.y))).type(torch.float)
    
    # Extract geometry parameters from plate_geometry_data
    # Format: [L, W, hole_radius, hole_x, hole_y, displacement]
    L = plate_geometry_data[i][0].item()
    W = plate_geometry_data[i][1].item()
    hole_radius = plate_geometry_data[i][2].item()
    hole_x = plate_geometry_data[i][3].item()
    hole_y = plate_geometry_data[i][4].item()
    displacement = plate_geometry_data[i][5].item()
    
    # Define the conditions for feature encoding
    node_positions = mesh_pos  # Use mesh_pos as node positions
    
    # NEW: Detect nodes on the hole edge
    # Calculate distance of each node from hole center
    delta_x = node_positions[:, 0] - hole_x
    delta_y = node_positions[:, 1] - hole_y
    distances_from_hole = torch.sqrt(delta_x**2 + delta_y**2)
    
    # Define nodes on hole edge (with a small tolerance)
    tolerance = 0.05 * hole_radius  # 5% of radius as tolerance
    is_hole_edge = torch.abs(distances_from_hole - hole_radius) < tolerance
    is_hole_edge = is_hole_edge.float().unsqueeze(1)
    
    # Define conditions for fixed and displaced nodes
    fixed_nodes = (node_positions[:, 0] <= 0.7)  # 0.7 is specific to this geometry
    loaded_nodes = (node_positions[:, 0] >= (node_positions[:, 0].max() - 1e-1))

    # Create feature vectors:
    # 1. is_fixed (binary indicator)
    is_fixed = torch.zeros(node_positions.shape[0], 1)
    is_fixed[fixed_nodes] = 1.0
    
    # 2. is_displaced (binary indicator)
    is_displaced = torch.zeros(node_positions.shape[0], 1)
    is_displaced[loaded_nodes] = 1.0
    
    # 3. displacement_amount (actual displacement value)
    displacement_amount = torch.zeros(node_positions.shape[0], 1)
    displacement_amount[loaded_nodes] = displacement
    
    # Concatenate all features: [x, y, is_fixed, is_displaced, displacement_amount]
    x = torch.cat((node_positions, is_hole_edge, is_fixed, is_displaced, displacement_amount), dim=1)
    
    # Add bidirectional edges if they don't already exist
    # First check if the graph already has bidirectional edges
    num_edges = edge_index.shape[1]
    edge_index_set = set((edge_index[0, i].item(), edge_index[1, i].item()) for i in range(num_edges))
    
    # Check if we need to add reverse edges
    needs_reverse_edges = False
    for j in range(num_edges):
        src, dst = edge_index[0, j].item(), edge_index[1, j].item()
        if (dst, src) not in edge_index_set:
            needs_reverse_edges = True
            break
    
    # Add reverse edges if needed
    if needs_reverse_edges:
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        # Concatenate original and reverse edges
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        
        # Also need to duplicate edge attributes for the reverse edges
        u_i = node_positions[edge_index[0, :num_edges]]  # Original edges' start nodes
        u_j = node_positions[edge_index[1, :num_edges]]  # Original edges' end nodes
        u_ij = u_i - u_j  # Calculate relative positions
        u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)  # Calculate norms
        edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).type(torch.float)
        
        # For reverse edges, the relative positions are negated
        u_ij_reverse = -u_ij  # Negated relative positions
        # Note: the norm stays the same
        edge_attr_reverse = torch.cat((u_ij_reverse, u_ij_norm), dim=-1).type(torch.float)
        
        # Combine original and reverse edge attributes
        edge_attr = torch.cat([edge_attr, edge_attr_reverse], dim=0)
    else:
        # Calculate edge attributes for existing edges
        u_i = node_positions[edge_index[0]]  # Get positions of start nodes
        u_j = node_positions[edge_index[1]]  # Get positions of end nodes
        u_ij = u_i - u_j  # Calculate relative positions
        u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)  # Calculate norms
        edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).type(torch.float)
    
    # Create the Data object with the new features and edges
    data_obj = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        cells=cells,
        mesh_pos=mesh_pos
    )

    data_list.append(data_obj)

# Save the processed data
processed_data_path = os.path.join('.', 'processed_data.pt')
torch.save(data_list, processed_data_path)
print(f"Data saved to {processed_data_path}")

# Only show shape summary once at the end
print("\nData summary:")
print(f"Number of samples: {len(data_list)}")
print(f"Node features: [x, y, is_fixed, is_displaced, displacement_amount]")
print(f"Node feature shape: {data_list[0].x.shape}")
print(f"Edge index shape: {data_list[0].edge_index.shape}")
print(f"Edge attribute shape: {data_list[0].edge_attr.shape}")

# Optional: Plot feature visualization for the first sample
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

sample_idx = 0
node_positions = data_list[sample_idx].x[:, :2]
features = ['Node Positions', 'Is Hole Edge', 'Is Fixed', 'Is Displaced', 'Displacement Amount', 'Strain (y)']

# Node positions plot
axes[0].scatter(node_positions[:, 0], node_positions[:, 1], c='blue', s=10)
axes[0].set_title(features[0])

# Plot is_hole_edge feature
sc = axes[1].scatter(node_positions[:, 0], node_positions[:, 1], 
                    c=data_list[sample_idx].x[:, 2].numpy(), cmap='RdYlGn', s=10)
axes[1].set_title(features[1])
fig.colorbar(sc, ax=axes[1])

# Plot is_fixed feature
sc = axes[2].scatter(node_positions[:, 0], node_positions[:, 1], 
                    c=data_list[sample_idx].x[:, 3].numpy(), cmap='cool', s=10)
axes[2].set_title(features[2])
fig.colorbar(sc, ax=axes[2])

# Plot is_displaced feature
sc = axes[3].scatter(node_positions[:, 0], node_positions[:, 1], 
                    c=data_list[sample_idx].x[:, 4].numpy(), cmap='autumn', s=10)
axes[3].set_title(features[3])
fig.colorbar(sc, ax=axes[3])

# Plot displacement_amount feature
sc = axes[4].scatter(node_positions[:, 0], node_positions[:, 1], 
                    c=data_list[sample_idx].x[:, 5].numpy(), cmap='viridis', s=10)
axes[4].set_title(features[4])
fig.colorbar(sc, ax=axes[4])

# Plot strain (output)
sc = axes[5].scatter(node_positions[:, 0], node_positions[:, 1], 
                    c=data_list[sample_idx].y.numpy().flatten(), cmap='plasma', s=10)
axes[5].set_title(features[5])
fig.colorbar(sc, ax=axes[5])

plt.tight_layout()
plt.savefig('feature_visualization.png')
print("Feature visualization saved as 'feature_visualization.png'")