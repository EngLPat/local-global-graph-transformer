"""
Data Utilities for FCLGA GraphTransformer.

This module provides functions for data normalization, statistics calculation,
and dataset analysis.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import torch


def normalize(to_normalize, mean_vec, std_vec):
    """
    Normalize tensor using provided mean and standard deviation.
    
    Args:
        to_normalize (torch.Tensor): Tensor to normalize.
        mean_vec (torch.Tensor): Mean values for normalization.
        std_vec (torch.Tensor): Standard deviation values for normalization.
    
    Returns:
        torch.Tensor: Normalized tensor.
    """
    # print(f"Shape of to_normalize before normalization: {to_normalize.shape}")
    # print(f"Shape of mean_vec: {mean_vec.shape}")
    # print(f"Shape of std_vec: {std_vec.shape}")
    normalized = (to_normalize - mean_vec) / std_vec
    # print(f"Shape of normalized: {normalized.shape}")
    return normalized


def unnormalize(to_unnormalize, mean_vec, std_vec):
    """
    Unnormalize tensor using provided mean and standard deviation.
    
    Args:
        to_unnormalize (torch.Tensor): Tensor to unnormalize.
        mean_vec (torch.Tensor): Mean values used in normalization.
        std_vec (torch.Tensor): Standard deviation values used in normalization.
    
    Returns:
        torch.Tensor: Unnormalized tensor.
    """
    return to_unnormalize * std_vec + mean_vec


def get_stats(data_list):
    """
    Calculate normalization statistics for graph dataset.
    
    Method for normalizing processed datasets. Given the processed data_list,
    calculates the mean and standard deviation for the node features, edge features,
    and node outputs.
    
    Args:
        data_list (list): List of PyTorch Geometric Data objects.
    
    Returns:
        list: List containing [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, 
              mean_vec_y, std_vec_y] for normalization.
    """
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


def analyze_node_features(dataset):
    """
    Analyze the structure of node features in the dataset.
    
    This function provides diagnostic information about the node features,
    useful for debugging and understanding the data structure.
    
    Args:
        dataset (list): List of PyTorch Geometric Data objects.
    """
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
