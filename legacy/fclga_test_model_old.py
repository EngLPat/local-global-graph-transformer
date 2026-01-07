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



## FOR optuna optimized model


import torch
import random
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, Dropout
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
import os
# import tensorflow.compat.v1 as tf
import math

# Import model and utilities from refactored modules
from src.models.fclga_graph_transformer import FCLGA_GraphTransformer
from src.models.processor_layer import ProcessorLayer
from src.models.attention import GlobalAttention
from src.utils.data_utils import normalize, unnormalize, get_stats, analyze_node_features
from src.utils.optimizer_utils import build_optimizer
from src.utils.visualization import plot_results, plot_regression, load_loss_data

# Import configuration
from config import paths as config_paths

# Define directories for datasets, checkpoints, and postprocessing (using config)
root_dir = config_paths.PROJECT_ROOT
dataset_dir = str(config_paths.DATASETS_DIR)
checkpoint_dir = str(config_paths.BEST_MODELS_DIR)
postprocess_dir = str(config_paths.ANIMATIONS_DIR)

print("dataset_dir {}".format(dataset_dir))


gnn_data_path = str(config_paths.DATASETS_DIR / 'processed_data.pt')
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

def plot_results_OLD(data, prediction, path, name, remote_stress=0.025):
    print('Generating strain fields...')
    
    # Ensure data is on CPU for matplotlib processing
    pos = data.mesh_pos.cpu().numpy()
    faces = data.cells.cpu().numpy()
    
    # Prepare the ground truth, prediction, error, and relative error data
    gs_strain = data.y[:, 0].cpu().numpy()
    pred_strain = prediction[:, 0].cpu().numpy()
    nominal_error = np.abs(pred_strain - gs_strain)  # Absolute nominal error
    relative_error_strain = np.abs((pred_strain - gs_strain) / remote_stress) * 100  # Relative error calculation

    # Print diagnostic information
    print(f"Ground truth range: {gs_strain.min():.6f} to {gs_strain.max():.6f}")
    print(f"Prediction range: {pred_strain.min():.6f} to {pred_strain.max():.6f}")
    print(f"Nominal error range: {nominal_error.min():.6f} to {nominal_error.max():.6f}")
    print(f"Relative error range: {relative_error_strain.min():.2f}% to {relative_error_strain.max():.2f}%")

    # Define titles with mathematical notation using LaTeX
    titles = [r'Actual Strain $\varepsilon_{xx}$', 
              r'Predicted Strain $\varepsilon_{xx}$', 
              r'Nominal Error $|\varepsilon_{xx}^{pred} - \varepsilon_{xx}^{act}|$', 
              'Relative Error (%)']
    
    # Save file names - CHANGED to .pdf
    file_names = ['actual_strain', 'predicted_strain', 'nominal_error', 'relative_error']
    
    strains = [gs_strain, pred_strain, nominal_error, relative_error_strain]
    
    # Calculate element-wise (cell-wise) values by averaging node values for each triangle
    # This matches what tripcolor does with shading='flat'
    gs_element_values = np.mean(gs_strain[faces], axis=1)
    pred_element_values = np.mean(pred_strain[faces], axis=1)
    
    min_strain = min(gs_element_values.min(), pred_element_values.min())
    max_strain = max(gs_element_values.max(), pred_element_values.max())
    
    print(f"Element-wise ground truth range: {gs_element_values.min():.6f} to {gs_element_values.max():.6f}")
    print(f"Element-wise prediction range: {pred_element_values.min():.6f} to {pred_element_values.max():.6f}")
    
    # Updated color limits - using actual element-wise data range for strain
    clim = [(min_strain, max_strain), (min_strain, max_strain), (0.0001, 0.0069), (0, 20)]
    colormaps = ['viridis', 'viridis', 'Reds', 'Reds']  # black to red for error

    # Create necessary directories
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save numerical data for later use
    data_path = os.path.join(path, name + '_data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    # Create separate plots folder
    plots_path = os.path.join(path, name + '_plots')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    
    # Save arrays for later analysis in other formats
    np.save(os.path.join(data_path, 'mesh_positions.npy'), pos)
    np.save(os.path.join(data_path, 'mesh_faces.npy'), faces)
    np.save(os.path.join(data_path, 'ground_truth.npy'), gs_strain)
    np.save(os.path.join(data_path, 'prediction.npy'), pred_strain)
    np.save(os.path.join(data_path, 'nominal_error.npy'), nominal_error)
    np.save(os.path.join(data_path, 'relative_error.npy'), relative_error_strain)
    
    # Save as CSV format for easy import to other software
    np.savetxt(os.path.join(data_path, 'results.csv'), 
               np.column_stack((pos, gs_strain, pred_strain, nominal_error, relative_error_strain)),
               delimiter=',', 
               header='x,y,ground_truth,prediction,nominal_error,relative_error')
    
    # Process each plot separately with black text and transparent background
    for i, (strain, title, clims, cmap, file_name) in enumerate(zip(strains, titles, clim, colormaps, file_names)):
        # Set up high-quality fonts with black text
        plt.rcParams.update({
            'text.usetex': True,  # Enable LaTeX
            'font.family': 'serif',
            'font.serif': ['CMU Serif', 'Computer Modern', 'serif'],
            'font.size': 11,
            'axes.labelsize': 11,
            'axes.titlesize': 11,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.titlesize': 11,
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'axes.titlecolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'svg.fonttype': 'none'  # SVG specific: embed fonts as text for better compatibility
        })
        
        # Create figure with transparent background - sized for journal publication
        fig = plt.figure(figsize=(4, 3), dpi=300)  # Compact size for individual plots
        fig.patch.set_alpha(0)  # Transparent figure background
        
        ax = plt.subplot(111)
        ax.set_aspect('equal')
        ax.set_axis_off()
        # ax.patch.set_alpha(0)  # Transparent axes background
        
        # Create the plot
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        mesh_plot = ax.tripcolor(triang, strain, shading='flat', cmap=cmap, vmin=clims[0], vmax=clims[1])
        ax.triplot(triang, 'ko-', ms=0.09, lw=0.21)  # Reduced node size by 40% (0.15->0.09) and edge width by 30% (0.3->0.21)
        # ax.set_title(title, fontsize=11, color='black')  # Remove title
        
        # Add colorbar only if not the first plot (since first two are the same scale)
        if i != 1:  # Skip colorbar for first plot (Actual Strain)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
            
            # Set 5 ticks on colorbar for better readability
            import matplotlib.ticker as ticker
            clb.locator = ticker.MaxNLocator(nbins=5)
            clb.update_ticks()
            
            # Make colorbar ticks and labels black
            clb.ax.yaxis.set_tick_params(color='black')
            plt.setp(plt.getp(clb.ax.axes, 'yticklabels'), color='black')
            clb.outline.set_edgecolor('black')
            
            # Set appropriate colorbar labels
            if 'Relative Error' in title:
                clb.set_label('% Error', color='black')

        # Save as SVG with transparent background
        plot_path = os.path.join(plots_path, f"{name}_{file_name}.pdf")
        plt.savefig(plot_path, format='pdf', bbox_inches='tight', transparent=True)
        
        # Close the figure to free memory
        plt.close(fig)
        print(f"Saved {plot_path}")
    
    
    # Also create separate comparison plots (instead of combined figure)
    plt.rcParams.update({
        'text.usetex': True,  # Enable LaTeX
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern', 'serif'],
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 11,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'svg.fonttype': 'none'  # SVG specific setting
    })
    
    # Use only first 3 plots (exclude relative error)
    combined_strains = strains[:3]
    combined_titles = titles[:3]
    combined_clim = clim[:3]
    combined_colormaps = colormaps[:3]
    combined_filenames = file_names[:3]
    
    # Create separate plots for comparison
    for i, (strain, title, clims, cmap, filename) in enumerate(zip(combined_strains, combined_titles, combined_clim, combined_colormaps, combined_filenames)):
        # Create individual figure with transparent background
        fig = plt.figure(figsize=(4, 3), dpi=300)  # Individual plot size
        fig.patch.set_alpha(0)  # Transparent figure background
        
        ax = plt.subplot(111)
        ax.set_aspect('equal')
        ax.set_axis_off()
        # ax.patch.set_alpha(0)  # Transparent axes background
        
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        mesh_plot = ax.tripcolor(triang, strain, shading='flat', cmap=cmap, vmin=clims[0], vmax=clims[1])
        ax.triplot(triang, 'ko-', ms=0.09, lw=0.21)  # Reduced node size by 40% (0.15->0.09) and edge width by 30% (0.3->0.21)
        # No title added
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')

        # Set 5 ticks on colorbar for better readability
        import matplotlib.ticker as ticker
        clb.locator = ticker.MaxNLocator(nbins=5)
        clb.update_ticks()

        # Make colorbar ticks and labels black
        clb.ax.yaxis.set_tick_params(color='black')
        plt.setp(plt.getp(clb.ax.axes, 'yticklabels'), color='black')
        clb.outline.set_edgecolor('black')

        # Set appropriate colorbar labels
        if 'Relative Error' in title:
            clb.set_label('% Error', color='black')
        
        # Save individual comparison plot as SVG
        comparison_path = os.path.join(path, f"{name}_comparison_{filename}.pdf")
        plt.savefig(comparison_path, format='pdf', bbox_inches='tight', transparent=True)
        plt.close(fig)
        print(f"Saved comparison plot: {comparison_path}")
    
    # Reset matplotlib settings to default before regression plot
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Create regression plot with black text and transparent background
    plot_regression(gs_strain, pred_strain, path, name, transparent=False, text_color='black')


def plot_regression_OLD(actual_strain, predicted_strain, path, name, transparent=False, text_color='black'):
    """Create a high-quality regression plot with customizable text color, filtering out padded nodes"""
    # Set up high-quality fonts with LaTeX
    plt.rcParams.update({
        'text.usetex': True,  # Enable LaTeX
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern', 'serif'],
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'axes.titlecolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'svg.fonttype': 'none'  # SVG specific: embed fonts as text
    })
    
    # Filter out padded nodes (where actual strain is very close to zero)
    # Use a small threshold instead of exact zero comparison
    threshold = 1e-6  # Adjust this threshold as needed
    non_zero_mask = np.abs(actual_strain) > threshold
    filtered_actual = actual_strain[non_zero_mask]
    filtered_predicted = predicted_strain[non_zero_mask]
    
    print(f"Original data points: {len(actual_strain)}")
    print(f"Filtered data points (actual strain > {threshold}): {len(filtered_actual)}")
    print(f"Removed {len(actual_strain) - len(filtered_actual)} padded/near-zero nodes from regression plot")
    print(f"Min actual strain in filtered data: {filtered_actual.min():.8f}")
    print(f"Max actual strain in filtered data: {filtered_actual.max():.8f}")
    
    # Check if we have any data left after filtering
    if len(filtered_actual) == 0:
        print("Warning: No non-zero actual strain values found. Skipping regression plot.")
        return
    
    # Calculate R-squared and RMSE on filtered data
    # R-squared calculation
    ss_res = np.sum((filtered_actual - filtered_predicted) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((filtered_actual - np.mean(filtered_actual)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    
    # RMSE calculation
    rmse = np.sqrt(np.mean((filtered_actual - filtered_predicted) ** 2))
    
    # Print metrics
    print(f"R-squared (R²): {r_squared:.4f}")
    print(f"RMSE: {rmse:.6f}")
    
        # Create regression plot with transparent background if requested
    fig_reg = plt.figure(figsize=(3.5, 3), dpi=300)
    fig_reg.patch.set_alpha(0)  # Transparent figure background

    ax_reg = fig_reg.add_subplot(111)
    ax_reg.set_facecolor('#D0D0D0')  # Light grey plot area background
    ax_reg.patch.set_alpha(1.0)  # Make axes background opaque
    
    # Create hexbin plot with filtered data - increased hexagon size
    hb = ax_reg.hexbin(
        filtered_actual.flatten(),
        filtered_predicted.flatten(),
        gridsize=35, cmap='Reds', mincnt=2, vmax=50  # Reduced gridsize for larger hexagons
    )
    
    # Add ideal prediction line using filtered data range
    actual_min = np.min(filtered_actual)
    actual_max = np.max(filtered_actual)
    ax_reg.plot(
        [actual_min, actual_max],
        [actual_min, actual_max],
        color=text_color, linestyle='--', linewidth=1.0, alpha=0.7, label="Ideal prediction"
    )
    
    # Set limits with a small margin
    margin = (actual_max - actual_min) * 0.05  # 5% margin
    ax_reg.set_xlim(left=max(0, actual_min - margin), right=actual_max + margin)
    ax_reg.set_ylim(bottom=max(0, actual_min - margin), top=actual_max + margin)
    
    # Reduce number of x-axis ticks to prevent overlap
    import matplotlib.ticker as ticker
    ax_reg.xaxis.set_major_locator(ticker.MaxNLocator(5))
    
    ax_reg.legend(loc="upper left")
    ax_reg.grid(False)  # Disable grid, use background color to show low-count points
    
    ax_reg.set_xlabel(r"Actual Strain $\varepsilon_{xx}$", color=text_color)
    ax_reg.set_ylabel(r"Predicted Strain $\varepsilon_{xx}^{\tiny\mathrm{FEA}}$", color=text_color)
    
    # Add text box with metrics in the plot
    textstr = f'$R^2$ = {r_squared:.4f}\nRMSE = {rmse:.6f}'
    props = dict(boxstyle='round', facecolor=(1,1,1,0.5) if transparent else 'white', alpha=0.8, edgecolor=text_color)
    ax_reg.text(0.95, 0.05, textstr, transform=ax_reg.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right', bbox=props, color=text_color)
    
    # Style the colorbar with matching text color
    cbar = fig_reg.colorbar(hb, ax=ax_reg)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    cbar.outline.set_edgecolor(text_color)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)
    cbar.set_label('Counts', color=text_color, size=11)
    
    plt.tight_layout()
    # When saving, use transparent=False so figure background stays transparent
    reg_path = os.path.join(path, name + '_regression.pdf')
    plt.savefig(reg_path, format='pdf', bbox_inches='tight', transparent=False)  # Changed back to True
    plt.close(fig_reg)
    print(f"Regression plot saved to {reg_path}")
    
    # Save metrics to a text file
    metrics_path = os.path.join(path, name + '_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Regression Metrics (Filtered Data)\n")
        f.write("===================================\n")
        f.write(f"Total data points: {len(actual_strain)}\n")
        f.write(f"Filtered data points: {len(filtered_actual)}\n")
        f.write(f"Threshold used: {threshold}\n")
        f.write(f"R-squared (R²): {r_squared:.6f}\n")
        f.write(f"RMSE: {rmse:.8f}\n")
        f.write(f"Min actual strain: {filtered_actual.min():.8f}\n")
        f.write(f"Max actual strain: {filtered_actual.max():.8f}\n")
        f.write(f"Mean actual strain: {np.mean(filtered_actual):.8f}\n")
        f.write(f"Mean predicted strain: {np.mean(filtered_predicted):.8f}\n")
    
    print(f"Metrics saved to {metrics_path}")
    
    # Reset matplotlib settings to default
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Return the calculated metrics for potential further use
    return r_squared, rmse, len(filtered_actual)


def load_loss_data_OLD(file_path):
    """Load loss data from a text file with format: epoch loss_value"""
    epochs = []
    losses = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    # Split the line into values
                    values = line.strip().split()
                    if len(values) >= 2:  # Ensure we have at least epoch and loss
                        epochs.append(float(values[0]))  # First value is epoch
                        losses.append(float(values[1]))  # Second value is loss
                    else:
                        # If there's only one value per line, assume it's just loss values
                        losses.append(float(values[0]))
                        epochs.append(len(epochs))  # Auto-increment epoch number
                        
        return epochs, losses
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [], []

def plot_epochs_OLD(train_loss_path, val_loss_path, output_dir, name="training_validation_loss", transparent=False, text_color='black'):
    """
    Create a high-quality plot of training and validation losses from files containing epoch-loss pairs.
    
    Parameters:
    -----------
    train_loss_path : str
        Path to the training loss text file
    val_loss_path : str
        Path to the validation loss text file
    output_dir : str
        Directory to save the output plot
    name : str
        Base name for the output file
    transparent : bool
        Whether to use a transparent background
    text_color : str
        Color to use for text and labels
    """
    import os
    import matplotlib.pyplot as plt
    
    # Load the loss data
    training_epochs, training_losses = load_loss_data(train_loss_path)
    validation_epochs, validation_losses = load_loss_data(val_loss_path)

    if not training_losses or not validation_losses:
        print("Training or validation loss file is missing or empty. Skipping loss plot.")
        return [], [], [], []
    
    # Print some statistics
    print(f"Training data: {len(training_losses)} epochs, min: {min(training_losses):.6f}, max: {max(training_losses):.6f}")
    print(f"Validation data: {len(validation_losses)} epochs, min: {min(validation_losses):.6f}, max: {max(validation_losses):.6f}")
    
    # Set up high-quality plot with black text
    plt.rcParams.update({
        'text.usetex': True,  # Enable LaTeX
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern', 'serif'],
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 11,
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'axes.titlecolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'svg.fonttype': 'none'  # SVG specific setting
    })
    
    # Create the figure with transparent background - optimized for journal layout
    fig = plt.figure(figsize=(5, 3), dpi=300)  # Moderate width for training curves
    if transparent:
        fig.patch.set_alpha(0)  # Transparent figure background
    
    ax = plt.subplot(111)
    # if transparent:
        # ax.patch.set_alpha(0)  # Transparent axes background
    
    # Plot the data
    ax.plot(training_epochs, training_losses, label="Training Loss", color="blue", linewidth=2)
    ax.plot(validation_epochs, validation_losses, label="Validation Loss", linestyle="--", color="darkorange", linewidth=2)
    
    # Add title and labels
    # ax.set_title("Model Training Progress", fontsize=11, color=text_color)  # Remove title
    ax.set_xlabel("Epochs", fontsize=11, color=text_color)
    ax.set_ylabel("Loss", fontsize=11, color=text_color)
    
    # Determine appropriate y-axis limits
    all_losses = training_losses + validation_losses
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    margin = (max_loss - min_loss) * 0.1  # 10% margin
    
    # Set axis limits with margin
    max_epoch = max(max(training_epochs), max(validation_epochs))
    ax.set_xlim(0, max_epoch + max_epoch * 0.05)  # Add 5% margin to x-axis
    ax.set_ylim(max(0, min_loss - margin), max_loss + margin)
    
    # Add legend with custom styling
    legend = ax.legend(loc="upper right", framealpha=0.7 if transparent else 1.0)
    for text in legend.get_texts():
        text.set_color(text_color)
    
    # Add grid
    ax.grid(True, color=text_color, alpha=0.3, linestyle='--')
    
    # Style the ticks
    ax.tick_params(colors=text_color, which='both')
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot as SVG
    plot_path = os.path.join(output_dir, f"{name}.pdf")
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', transparent=transparent)
    plt.close(fig)
    print(f"Loss plot saved to {plot_path}")
    
    return training_epochs, training_losses, validation_epochs, validation_losses

file_path = str(config_paths.DATASETS_DIR / 'processed_data.pt')
dataset_full_timesteps = torch.load(gnn_data_path, weights_only=False)
dataset = torch.load(file_path, weights_only=False)
if not isinstance(dataset, list):
    dataset = [dataset]
dataset = dataset[:1]

print(dataset)
len(dataset_full_timesteps)/5

# normalize, unnormalize, get_stats, build_optimizer, analyze_node_features
# are now imported from utils package

def normalize_OLD(to_normalize, mean_vec, std_vec):
    # print(f"Shape of to_normalize before normalization: {to_normalize.shape}")
    # print(f"Shape of mean_vec: {mean_vec.shape}")
    # print(f"Shape of std_vec: {std_vec.shape}")
    normalized = (to_normalize - mean_vec) / std_vec
    # print(f"Shape of normalized: {normalized.shape}")
    return normalized

def unnormalize_OLD(to_unnormalize,mean_vec,std_vec):
    return to_unnormalize*std_vec+mean_vec

def get_stats_OLD(data_list):
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

# Model classes are now imported from models package

class GlobalAttention_OLD(torch.nn.Module):
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

class MeshGraphNet_OLD(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, args, emb=False):
        super(MeshGraphNet_OLD, self).__init__()
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
        self.dropout_rate = getattr(args, 'dropout_rate', 0.38)  # Use args.dropout_rate if available

        self.attention_freq = getattr(args, 'attention_freq', 
                                    max(1, self.num_layers // 2) if self.num_layers <= 8 else 8)

        # Add skip connection projection
        # self.skip_projection = Linear(hidden_dim, hidden_dim)

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

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension
        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # Step 2: perform message passing with latent node/edge embeddings
        # layer_outputs = [x]  # Store layer outputs for skip connections
        for i in range(self.num_layers):
            # Add skip connection from 8 layers back (keep existing logic)
            # if i >= 8 and i % 8 == 0:  # Every 8 layers after the 8th
            #     x = x + self.skip_projection(layer_outputs[i-8])
                
            
            # FIXED: Adaptive global attention frequency
            if i % self.attention_freq == 0: 
                # Create batch index if not provided
                batch = getattr(data, 'batch', None)
                if batch is None:
                    batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
                    
                global_info = self.global_attention(x, batch=batch)
                x = x + 0.2 * global_info  # Mix with local representations

            x, edge_attr = self.processor[i](x, edge_index, edge_attr)
            # layer_outputs.append(x)  # Store current layer output

        # step 3: decode latent node embeddings into physical quantities of interest
        return self.decoder(x)

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
    
class ProcessorLayer_OLD(MessagePassing):
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
    
def build_optimizer_OLD(args, params):
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

def analyze_node_features_OLD(dataset):
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

def test(loader, device, test_model, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, is_validation, args, delta_t=0.01, save_model_preds=False, model_type=None):
    '''
    Calculates test set losses and validation set errors with proper masking.
    '''
    loss = 0
    velo_rmse = 0
    num_loops = 0

    # For overall RMSE
    all_preds = []
    all_gts = []
    sample_rmses = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss += test_model.loss(pred, data, mean_vec_y, std_vec_y)

            if is_validation:
                # Unnormalize predictions and ground truth
                eval_y = unnormalize(pred, mean_vec_y, std_vec_y)
                gs_y = data.y

                # Flatten and accumulate for overall RMSE
                all_preds.append(eval_y.cpu().numpy().flatten())
                all_gts.append(gs_y.cpu().numpy().flatten())

                # **MASKING STRATEGY FROM SQUARE FILE**
                threshold = 1e-6
                non_zero_mask = torch.abs(gs_y) > threshold  # Same as square file
                
                if non_zero_mask.sum() > 0:
                    filtered_eval_y = eval_y[non_zero_mask]
                    filtered_gs_y = gs_y[non_zero_mask]
                    sample_rmse = torch.sqrt(torch.mean((filtered_eval_y - filtered_gs_y) ** 2)).item()
                    print(f"Sample {num_loops} RMSE (filtered, {non_zero_mask.sum().item()} nodes): {sample_rmse:.6f}")
                    sample_rmses.append(sample_rmse)
                    velo_rmse += sample_rmse  # Accumulate for average
                else:
                    print(f"Sample {num_loops}: No non-zero nodes found")
                    sample_rmses.append(0.0)

        num_loops += 1

    # Convert to numpy
    sample_rmses = np.array(sample_rmses)
    
    # Overall RMSE calculation
    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)

    # **SAME MASKING FOR OVERALL RMSE**
    threshold = 1e-6
    non_zero_mask = np.abs(all_gts) > threshold
    filtered_all_preds = all_preds[non_zero_mask]
    filtered_all_gts = all_gts[non_zero_mask]

    overall_rmse = np.sqrt(np.mean((filtered_all_preds - filtered_all_gts) ** 2))
    print(f"\nOverall RMSE across all samples (filtered, {non_zero_mask.sum()} nodes): {overall_rmse:.6f}")
    print(f"Removed {len(all_gts) - non_zero_mask.sum()} padded/near-zero nodes")

    # Per-sample statistics
    if len(sample_rmses) > 0:
        print("\nRMSE Statistics:")
        print(f"  Mean: {np.mean(sample_rmses):.6f}")
        print(f"  Median: {np.median(sample_rmses):.6f}")
        print(f"  Std: {np.std(sample_rmses):.6f}")
        print(f"  Min: {np.min(sample_rmses):.6f}")
        print(f"  Max: {np.max(sample_rmses):.6f}")

    return loss / num_loops, overall_rmse, sample_rmses

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

    model = FCLGA_GraphTransformer(num_node_features, num_edge_features, 
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
            PATH = str(config_paths.BEST_MODELS_DIR / f"{model_name}.pt")
            torch.save(best_model.state_dict(), PATH)
    
    # Save final dataframe
    PATH = str(config_paths.BEST_MODELS_DIR / f"{model_name}.csv")
    df.to_csv(PATH, index=False)
    
    # Plot results from final model
    plot_name = 'final_epoch_results'
    visualize(val_loader, best_model, postprocess_dir, plot_name, stats_list)
    
    return val_losses, losses, velo_val_losses, best_model

def save_plots(args, train_losses, val_losses, test_losses=None, velo_val_losses=None):
    model_name = 'model_nl' + str(args.num_layers) + '_bs' + str(args.batch_size) + \
                 '_hd' + str(args.hidden_dim) + '_ep' + str(args.epochs) + '_wd' + str(args.weight_decay) + \
                 '_lr' + str(args.lr) + '_shuff_' + str(args.shuffle) + '_tr' + str(args.train_size) + '_te' + str(args.test_size)

    if not os.path.isdir(args.postprocess_dir):
        os.mkdir(args.postprocess_dir)

    # CHANGED: Save as SVG instead of PDF
    PATH = os.path.join(args.postprocess_dir, model_name + '.pdf')

    # Set SVG specific matplotlib parameters
    plt.rcParams.update({
        'svg.fonttype': 'none'  # Embed fonts as text for better compatibility
    })

    plt.figure(figsize=(10, 6))
    # plt.title('Losses Plot')  # Remove title
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
    # Save as SVG instead of PDF
    plt.savefig(PATH, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Plot saved at: {PATH}")
    
def evaluate_final_model(test_dataset, best_model, device, stats_list, args):
    """
    Evaluate the final model on the test set after training is complete.
    Calculates mean RMSE across all test samples with proper masking.
    """
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch size 1 for per-sample RMSE
    
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(device), std_vec_x.to(device), mean_vec_edge.to(device), 
        std_vec_edge.to(device), mean_vec_y.to(device), std_vec_y.to(device))
    
    best_model.eval()
    
    print(f"\n{'='*70}")
    print("EVALUATING TEST DATASET - INDIVIDUAL SAMPLE RMSEs")
    print(f"{'='*70}\n")
    
    total_loss = 0
    all_preds = []
    all_gts = []
    individual_rmses = []
    num_samples = 0
    
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            # Get prediction
            pred = best_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            
            # Calculate loss
            loss = best_model.loss(pred, data, mean_vec_y, std_vec_y)
            total_loss += loss.item()
            
            # Unnormalize for RMSE calculation
            eval_y = unnormalize(pred, mean_vec_y, std_vec_y)
            gs_y = data.y
            
            # Flatten and accumulate for global RMSE
            all_preds.append(eval_y.cpu().numpy().flatten())
            all_gts.append(gs_y.cpu().numpy().flatten())
            
            # **SAME MASKING AS IN test() FUNCTION**
            threshold = 1e-6
            non_zero_mask = torch.abs(gs_y) > threshold
            
            if non_zero_mask.sum() > 0:
                filtered_eval_y = eval_y[non_zero_mask]
                filtered_gs_y = gs_y[non_zero_mask]
                sample_rmse = torch.sqrt(torch.mean((filtered_eval_y - filtered_gs_y) ** 2)).item()
                individual_rmses.append(sample_rmse)
                
                # Print per-sample RMSE
                print(f"Sample {num_samples:3d}: RMSE = {sample_rmse:.8f} "
                      f"({non_zero_mask.sum().item():5d} non-zero nodes, "
                      f"{(~non_zero_mask).sum().item():4d} masked)")
            else:
                print(f"Sample {num_samples:3d}: No non-zero nodes found (all masked)")
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
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Number of test samples: {num_samples}")
    print(f"Mean Test Loss: {mean_loss:.5f}")
    print("\n--- RMSE METRICS ---")
    print(f"Global RMSE (all nodes, all samples): {global_rmse:.8f}")
    print(f"Mean of per-sample RMSEs: {mean_sample_rmse:.8f}")
    print(f"RMSE Std Dev: {rmse_std:.8f}")
    print(f"RMSE Range: [{rmse_min:.8f}, {rmse_max:.8f}]")
    print("\n--- MASKING INFO ---")
    print(f"Total nodes across all samples: {len(all_gts)}")
    print(f"Non-zero nodes (used): {non_zero_mask_global.sum()}")
    print(f"Masked nodes (padded/zero): {(~non_zero_mask_global).sum()}")
    print(f"Masking percentage: {((~non_zero_mask_global).sum() / len(all_gts) * 100):.2f}%")
    print(f"{'='*70}\n")
    
    # Save detailed results
    results_path = os.path.join(args.postprocess_dir, 'test_results_detailed.txt')
    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Number of test samples: {num_samples}\n")
        f.write(f"Mean Test Loss: {mean_loss:.5f}\n\n")
        f.write("--- RMSE METRICS ---\n")
        f.write(f"Global RMSE (all nodes, all samples): {global_rmse:.8f}\n")
        f.write(f"Mean of per-sample RMSEs: {mean_sample_rmse:.8f}\n")
        f.write(f"RMSE Std Dev: {rmse_std:.8f}\n")
        f.write(f"RMSE Range: [{rmse_min:.8f}, {rmse_max:.8f}]\n\n")
        f.write("--- MASKING INFO ---\n")
        f.write(f"Total nodes across all samples: {len(all_gts)}\n")
        f.write(f"Non-zero nodes (used): {non_zero_mask_global.sum()}\n")
        f.write(f"Masked nodes (padded/zero): {(~non_zero_mask_global).sum()}\n")
        f.write(f"Masking percentage: {((~non_zero_mask_global).sum() / len(all_gts) * 100):.2f}%\n\n")
        f.write("Individual Sample RMSEs:\n")
        for i, rmse in enumerate(individual_rmses):
            f.write(f"  Sample {i}: {rmse:.8f}\n")
    
    print(f"Detailed results saved to: {results_path}")
    
    # Visualize a representative sample (closest to mean RMSE)
    closest_idx = np.argmin(np.abs(individual_rmses - mean_sample_rmse))
    plot_name = f'test_representative_sample_{closest_idx}'
    single_loader = DataLoader([test_dataset[closest_idx]], batch_size=1, shuffle=False)
    visualize(single_loader, best_model, args.postprocess_dir, plot_name, stats_list)
    
    return mean_loss, global_rmse

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

best_params = {
    'model_type': 'meshgraphnet',
    'num_layers': 5,
    'attention_freq': 4,
    'batch_size': 4,
    'hidden_dim': 128,
    'dropout_rate': 0.21887774707222715,
    'epochs': 3000,  # Set as needed
    'opt': 'adam',
    'opt_scheduler': 'step',  # Or as used in your code
    'opt_decay_step': 55,
    'opt_decay_rate': 0.701057102063354,
    'opt_restart': 0,
    'weight_decay': 9.727199495994628e-05,
    'lr': 0.0003884856764273311,
    'train_size': 400,  # Set as needed
    'test_size': 100,   # Set as needed
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'shuffle': True,
    'save_velo_val': True,
    'save_best_model': True,
    'checkpoint_dir': checkpoint_dir,
    'postprocess_dir': postprocess_dir
}
args = objectview(best_params)  # Fixed: should be best_params, not args

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


# Add this function after the verify_physical_consistency function and before the analyze_node_features call

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
    mean_vec_x, std_vec_x = mean_vec_x.to(device), std_vec_x.to(device)
    mean_vec_edge, std_vec_edge = mean_vec_edge.to(device), std_vec_edge.to(device)
    mean_vec_y, std_vec_y = mean_vec_y.to(device), std_vec_y.to(device)
    
    # Prepare single sample for inference
    sample = test_dataset[0].to(device)
    model.eval()
    
    # Warm up GPU
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
    
    # Time GNN inference
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            model(sample, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)

    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate average inference time
    total_inference_time = end_time - start_time
    avg_inference_time = total_inference_time / num_runs
    
    # Actual FEM simulation time from Abaqus
    actual_fem_time = 9.0  # 9 seconds from your Abaqus simulation
    
    results = {
        'gnn_inference_time': avg_inference_time,
        'num_runs': num_runs,
        'total_time': total_inference_time,
        'fem_time': actual_fem_time,
        'speedup': actual_fem_time / avg_inference_time,
        'time_saved': actual_fem_time - avg_inference_time
    }
    
    print("\nGNN Inference Results:")
    print(f"  Average inference time: {avg_inference_time*1000:.3f} ms ({avg_inference_time:.6f} s)")
    print(f"  Total time for {num_runs} runs: {total_inference_time:.3f} seconds")
    print(f"  Inference frequency: {1/avg_inference_time:.1f} predictions/second")
    
    print("\nSpeedup Analysis vs FEM:")
    print(f"  Abaqus FEM time: {actual_fem_time:.1f} seconds")
    print(f"  GNN inference time: {avg_inference_time*1000:.3f} ms")
    print(f"  Speedup: {results['speedup']:.0f}× faster than FEM")
    print(f"  Time saved per prediction: {results['time_saved']:.3f} seconds")
    
    # Break-even analysis for training cost
    training_time_estimate = 3600  # 1 hour estimate - update with your actual training time
    time_saved_per_pred = results['time_saved']
    breakeven_predictions = training_time_estimate / time_saved_per_pred if time_saved_per_pred > 0 else float('inf')
    
    print("\nBreak-even Analysis:")
    print(f"  Training time estimate: {training_time_estimate/3600:.1f} hours")
    print(f"  Break-even point: {breakeven_predictions:.0f} predictions")
    print(f"  Training ROI: Profitable after {breakeven_predictions:.0f} simulations")
    
    # Save results to file
    benchmark_path = str(config_paths.ANIMATIONS_DIR / 'inference_benchmark.txt')
    with open(benchmark_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("INFERENCE TIME BENCHMARK\n")
        f.write("="*60 + "\n\n")
        f.write("GNN Inference Results:\n")
        f.write(f"  Average inference time: {avg_inference_time*1000:.3f} ms ({avg_inference_time:.6f} s)\n")
        f.write(f"  Number of runs: {num_runs}\n")
        f.write(f"  Total time: {total_inference_time:.3f} seconds\n")
        f.write(f"  Inference frequency: {1/avg_inference_time:.1f} predictions/second\n\n")
        f.write("Speedup Analysis vs FEM:\n")
        f.write(f"  Abaqus FEM time: {actual_fem_time:.1f} seconds\n")
        f.write(f"  GNN inference time: {avg_inference_time*1000:.3f} ms\n")
        f.write(f"  Speedup: {results['speedup']:.0f}× faster\n")
        f.write(f"  Time saved per prediction: {results['time_saved']:.3f} seconds\n\n")
        f.write("Break-even Analysis:\n")
        f.write(f"  Training time estimate: {training_time_estimate/3600:.1f} hours\n")
        f.write(f"  Break-even point: {breakeven_predictions:.0f} predictions\n")
        f.write(f"  Training ROI: Profitable after {breakeven_predictions:.0f} simulations\n")
    
    print(f"\nBenchmark results saved to: {benchmark_path}")
    print("="*60 + "\n")
    
    return results

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

#load best model

num_node_features = test_dataset[0].x.shape[1]
num_edge_features = test_dataset[0].edge_attr.shape[1]
num_classes = 1
best_model = FCLGA_GraphTransformer(num_node_features, num_edge_features, 
                        args.hidden_dim, num_classes, args).to(device)

# Load the saved model weights
model_path = 'results/0_standard_20251119_181333/best_models/model_nl5_bs4_hd128_ep3000_wd9.727199495994628e-05_lr0.0003884856764273311_shuff_True.pt'
best_model.load_state_dict(torch.load(model_path, map_location=device))
best_model.eval()  # Set model to evaluation mode
print(f"Loaded model from: {model_path}")

benchmark_results = benchmark_inference_time(best_model, test_dataset, device, stats_list, num_runs=100)

# Final evaluation on test set
final_test_loss, final_test_rmse = evaluate_final_model(test_dataset, best_model, device, stats_list, args)
print(f"Final Test Loss: {final_test_loss:.5f}")
print(f"Final Test RMSE: {final_test_rmse:.5f}")

# Plot losses (since we're only testing, we don't have training/validation losses)
test_losses = [final_test_loss]
save_plots(args, [0], [0], test_losses)  # Pass empty lists for train/val losses

# Generate additional visualizations on test samples
for i in range(min(25, len(test_dataset))):  # Visualize first 5 test samples
    plot_name = f'test_sample_{i}_results'
    single_loader = DataLoader([test_dataset[i]], batch_size=1, shuffle=False)
    visualize(single_loader, best_model, args.postprocess_dir, plot_name, stats_list)

# plot_epochs(
#     train_loss_path="./training_loss.txt", 
#     val_loss_path="./val_loss.txt", 
#     output_dir=postprocess_dir,
#     name="model_training_history",
#     transparent=True,
#     text_color='black'
# )