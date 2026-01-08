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

# Define directories for datasets, checkpoints, and postprocessing
root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir, 'datasets')
checkpoint_dir = os.path.join(root_dir, 'best_models')
postprocess_dir = os.path.join(root_dir, 'animations')

print("dataset_dir {}".format(dataset_dir))


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

def plot_results(data, prediction, path, name, remote_stress=0.025):
    print('Generating strain fields...')
    
    # Ensure data is on CPU for matplotlib processing
    pos = data.mesh_pos.cpu().numpy()
    faces = data.cells.cpu().numpy()
    
    # Prepare the ground truth, prediction, error, and relative error data
    gs_strain = data.y[:, 0].cpu().numpy()
    pred_strain = prediction[:, 0].cpu().numpy()
    nominal_error = np.abs(pred_strain - gs_strain)  # Absolute nominal error
    epsilon = 1e-10
    # relative_error_strain = ((pred_strain - gs_strain) / remote_stress) * 100  # Relative error calculation
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
    
    # Save file names - CHANGED to .svg
    file_names = ['actual_strain', 'predicted_strain', 'nominal_error', 'relative_error']
    
    strains = [gs_strain, pred_strain, nominal_error, relative_error_strain]
    max_strain = max(gs_strain.max(), pred_strain.max())
    max_error = nominal_error.max()
    
    # Updated color limits - using black to red for nominal error
    clim = [(0, max_strain), (0, max_strain), (0, 0.004), (-20, 20)]
    colormaps = ['viridis', 'viridis', 'Reds', 'coolwarm']  # black to red for error

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
        ax.patch.set_alpha(0)  # Transparent axes background
        
        # Create the plot
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        mesh_plot = ax.tripcolor(triang, strain, shading='flat', cmap=cmap, vmin=clims[0], vmax=clims[1])
        ax.triplot(triang, 'ko-', ms=0.15, lw=0.3)  # Reduced node size by another 25% (0.2 -> 0.15)
        ax.set_title(title, fontsize=11, color='black')
        
        # Add colorbar only if not the first plot (since first two are the same scale)
        if i != 0:  # Skip colorbar for first plot (Actual Strain)
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
        plt.savefig(plot_path, format='pdf', bbox_inches='tight', transparent=False)
        
        # Close the figure to free memory
        plt.close(fig)
        print(f"Saved {plot_path}")
    
    # NEW: Create 2x2 subplot layout with requested arrangement
    # Set up high-quality fonts with black text
    plt.rcParams.update({
        'text.usetex': True,  # Enable LaTeX
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern', 'serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 10,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'svg.fonttype': 'none'
    })
    
    # Create 2x2 subplot figure - sized for journal column width
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 6), dpi=300)
    fig.patch.set_alpha(0)  # Transparent figure background
    
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Subplot 1,1: Actual Strain
    ax1.set_aspect('equal')
    ax1.set_axis_off()
    ax1.patch.set_alpha(0)
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    mesh_plot1 = ax1.tripcolor(triang, gs_strain, shading='flat', cmap='viridis', vmin=0, vmax=max_strain)
    ax1.triplot(triang, 'ko-', ms=0.1, lw=0.2)
    ax1.set_title(r'(a) Actual Strain $\varepsilon_{xx}$', fontsize=10, color='black', pad=10)
    
    # Add colorbar for subplot 1
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    clb1 = fig.colorbar(mesh_plot1, cax=cax1, orientation='vertical')
    clb1.locator = ticker.MaxNLocator(nbins=4)
    clb1.update_ticks()
    clb1.ax.yaxis.set_tick_params(color='black', labelsize=8)
    plt.setp(plt.getp(clb1.ax.axes, 'yticklabels'), color='black')
    clb1.outline.set_edgecolor('black')
    
    # Subplot 1,2: Predicted Strain
    ax2.set_aspect('equal')
    ax2.set_axis_off()
    ax2.patch.set_alpha(0)
    mesh_plot2 = ax2.tripcolor(triang, pred_strain, shading='flat', cmap='viridis', vmin=0, vmax=max_strain)
    ax2.triplot(triang, 'ko-', ms=0.1, lw=0.2)
    ax2.set_title(r'(b) Predicted Strain $\varepsilon_{xx}$', fontsize=10, color='black', pad=10)
    
    # Add colorbar for subplot 2
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    clb2 = fig.colorbar(mesh_plot2, cax=cax2, orientation='vertical')
    clb2.locator = ticker.MaxNLocator(nbins=4)
    clb2.update_ticks()
    clb2.ax.yaxis.set_tick_params(color='black', labelsize=8)
    plt.setp(plt.getp(clb2.ax.axes, 'yticklabels'), color='black')
    clb2.outline.set_edgecolor('black')
    
    # Subplot 2,1: Nominal Error
    ax3.set_aspect('equal')
    ax3.set_axis_off()
    ax3.patch.set_alpha(0)
    mesh_plot3 = ax3.tripcolor(triang, nominal_error, shading='flat', cmap='Reds', vmin=0, vmax=0.004)
    ax3.triplot(triang, 'ko-', ms=0.1, lw=0.2)
    ax3.set_title(r'(c) Nominal Error $|\varepsilon_{xx}^{pred} - \varepsilon_{xx}^{act}|$', fontsize=10, color='black', pad=10)
    
    # Add colorbar for subplot 3
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    clb3 = fig.colorbar(mesh_plot3, cax=cax3, orientation='vertical')
    clb3.locator = ticker.MaxNLocator(nbins=4)
    clb3.update_ticks()
    clb3.ax.yaxis.set_tick_params(color='black', labelsize=8)
    plt.setp(plt.getp(clb3.ax.axes, 'yticklabels'), color='black')
    clb3.outline.set_edgecolor('black')
    
    # Subplot 2,2: Regression plot
    ax4.patch.set_alpha(0)
    
    # Filter out padded nodes for regression plot
    threshold = 1e-6
    non_zero_mask = np.abs(gs_strain) > threshold
    filtered_actual = gs_strain[non_zero_mask]
    filtered_predicted = pred_strain[non_zero_mask]
    
    # Calculate R-squared and RMSE
    ss_res = np.sum((filtered_actual - filtered_predicted) ** 2)
    ss_tot = np.sum((filtered_actual - np.mean(filtered_actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((filtered_actual - filtered_predicted) ** 2))
    
    # Create hexbin plot
    hb = ax4.hexbin(
        filtered_actual.flatten(),
        filtered_predicted.flatten(),
        gridsize=25, cmap='Reds', mincnt=2, vmax=50
    )
    
    # Add ideal prediction line
    actual_min = np.min(filtered_actual)
    actual_max = np.max(filtered_actual)
    ax4.plot([actual_min, actual_max], [actual_min, actual_max], 
             color='black', linestyle='--', linewidth=1.0, alpha=0.7, label="Ideal prediction")
    
    # Set limits
    margin = (actual_max - actual_min) * 0.05
    ax4.set_xlim(left=max(0, actual_min - margin), right=actual_max + margin)
    ax4.set_ylim(bottom=max(0, actual_min - margin), top=actual_max + margin)
    
    ax4.set_xlabel(r"Actual Strain $\varepsilon_{xx}$", fontsize=10, color='black')
    ax4.set_ylabel(r"Predicted Strain $\varepsilon_{xx}$", fontsize=10, color='black')
    ax4.set_title(r'(d) Regression', fontsize=10, color='black', pad=10)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, color='black', alpha=0.3)
    ax4.tick_params(colors='black', labelsize=8)
    
    # Add metrics text box
    textstr = f'$R^2$ = {r_squared:.3f}\nRMSE = {rmse:.4f}'
    props = dict(boxstyle='round', facecolor=(1,1,1,0.7), alpha=0.8, edgecolor='black')
    ax4.text(0.95, 0.05, textstr, transform=ax4.transAxes, fontsize=8,
             verticalalignment='bottom', horizontalalignment='right', bbox=props, color='black')
    
    # Add colorbar for regression hexbin
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    cbar4 = fig.colorbar(hb, cax=cax4)
    cbar4.ax.yaxis.set_tick_params(color='black', labelsize=8)
    cbar4.outline.set_edgecolor('black')
    plt.setp(plt.getp(cbar4.ax.axes, 'yticklabels'), color='black')
    cbar4.set_label('Counts', color='black', size=8)
    
    # Save the 2x2 subplot layout
    subplot_path = os.path.join(path, name + '_2x2_layout.pdf')
    plt.savefig(subplot_path, format='pdf', bbox_inches='tight', transparent=False)
    plt.close(fig)
    print(f"Saved 2x2 layout: {subplot_path}")
    
    # Also create a combined figure for comparison (existing code)
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
    
    # Create combined figure with transparent background - wide format for comparison (without relative error)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), dpi=300)  # 3-panel comparison without relative error
    fig.patch.set_alpha(0)  # Transparent figure background
    
    # Increase spacing between subplots
    plt.subplots_adjust(wspace=0.3)  # Increased spacing between plots
    
    # Use only first 3 plots (exclude relative error)
    combined_strains = strains[:3]
    combined_titles = titles[:3]
    combined_clim = clim[:3]
    combined_colormaps = colormaps[:3]
    
    for ax, strain, title, clims, cmap in zip(axes, combined_strains, combined_titles, combined_clim, combined_colormaps):
        ax.cla()
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.patch.set_alpha(0)  # Transparent axes background
        
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        mesh_plot = ax.tripcolor(triang, strain, shading='flat', cmap=cmap, vmin=clims[0], vmax=clims[1])
        ax.triplot(triang, 'ko-', ms=0.15, lw=0.3)  # Reduced node size by another 25% (0.2 -> 0.15)
        ax.set_title(title, fontsize=11, color='black')
        
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
    
    # Save the combined plot as SVG
    combined_path = os.path.join(path, name + '_comparison.pdf')
    plt.savefig(combined_path, format='pdf', bbox_inches='tight', transparent=False)
    plt.close(fig)
    print(f"Saved combined plot: {combined_path}")
    
    # Reset matplotlib settings to default before regression plot
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Create regression plot with black text and transparent background
    plot_regression(gs_strain, pred_strain, path, name, transparent=False, text_color='black')

def plot_regression(actual_strain, predicted_strain, path, name, transparent=False, text_color='black'):
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
    fig_reg = plt.figure(figsize=(3.5, 3), dpi=300)  # Square format for regression plot
    if transparent:
        fig_reg.patch.set_alpha(0)  # Transparent background
    else:
        fig_reg.patch.set_facecolor('white')  # White background
        
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
    ax_reg.grid(True, color=text_color, alpha=0.3)
    
    # Updated title to include metrics
    title_text = f"Regression"
    ax_reg.set_title(title_text, fontsize=11, color=text_color)
    ax_reg.set_xlabel(r"Actual Strain $\varepsilon_{xx}^{\mathrm{FEA}}$", color=text_color)
    ax_reg.set_ylabel(r"Predicted Strain $\varepsilon_{xx}$", color=text_color)
    
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
    # Save as SVG instead of PNG
    reg_path = os.path.join(path, name + '_regression.pdf')
    plt.savefig(reg_path, format='pdf', bbox_inches='tight', transparent=transparent)
    plt.close(fig_reg)
    print(f"Regression plot saved to {reg_path}")
    
    # Save metrics to a text file
    metrics_path = os.path.join(path, name + '_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Regression Metrics (Filtered Data)\n")
        f.write(f"===================================\n")
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


def load_loss_data(file_path):
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

def plot_epochs(train_loss_path, val_loss_path, output_dir, name="training_validation_loss", transparent=False, text_color='black'):
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
    import numpy as np
    
    # Load the loss data
    training_epochs, training_losses = load_loss_data(train_loss_path)
    validation_epochs, validation_losses = load_loss_data(val_loss_path)
    
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
    if transparent:
        ax.patch.set_alpha(0)  # Transparent axes background
    
    # Plot the data
    ax.plot(training_epochs, training_losses, label="Training Loss", color="blue", linewidth=2)
    ax.plot(validation_epochs, validation_losses, label="Validation Loss", linestyle="--", color="darkorange", linewidth=2)
    
    # Add title and labels
    ax.set_title("Model Training Progress", fontsize=11, color=text_color)
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
        self.dropout_rate = getattr(args, 'dropout_rate', 0.38)  # Use args.dropout_rate if available

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

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension
        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # Step 2: perform message passing with latent node/edge embeddings
        layer_outputs = [x]  # Store layer outputs for skip connections
        for i in range(self.num_layers):
            # Add skip connection from 8 layers back (keep existing logic)
            if i >= 8 and i % 8 == 0:  # Every 8 layers after the 8th
                x = x + self.skip_projection(layer_outputs[i-8])
                
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)
            layer_outputs.append(x)  # Store current layer output
            
            # FIXED: Adaptive global attention frequency
            if (i + 1) % self.attention_freq == 0:  # Use adaptive frequency
                # Create batch index if not provided
                batch = getattr(data, 'batch', None)
                if batch is None:
                    batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
                    
                global_info = self.global_attention(x, batch=batch)
                x = x + 0.2 * global_info  # Mix with local representations

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

def test(loader, device, test_model, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, is_validation, args, delta_t=0.01, save_model_preds=False, model_type=None):
    '''
    Calculates test set losses and validation set errors.
    '''
    loss = 0
    velo_rmse = 0
    num_loops = 0

    # For overall RMSE
    all_preds = []
    all_gts = []
    sample_rmses = []  # Initialize list to store per-sample RMSEs

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            # Calculate the loss for the model given the test set

            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)

            if device == 'cuda':
                torch.cuda.synchronize()    
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            print(f"Inference time for this sample: {inference_time*1000:.3f} ms")

            loss += test_model.loss(pred, data, mean_vec_y, std_vec_y)

            # Calculate validation error if asked to
            if is_validation:
                # Unnormalize the predictions and ground truth
                eval_y = unnormalize(pred, mean_vec_y, std_vec_y)
                gs_y = data.y

                # Flatten and accumulate
                all_preds.append(eval_y.cpu().numpy().flatten())
                all_gts.append(gs_y.cpu().numpy().flatten())

                # Calculate the error
                error = torch.sum((eval_y - gs_y) ** 2, axis=1)
                velo_rmse += torch.sqrt(torch.mean(error))
                # Calculate and print per-sample RMSE
                threshold = 1e-6
                non_zero_mask = torch.abs(gs_y) > threshold
                if non_zero_mask.sum() > 0:  # Ensure we have non-zero nodes
                    filtered_eval_y = eval_y[non_zero_mask]
                    filtered_gs_y = gs_y[non_zero_mask]
                    sample_rmse = torch.sqrt(torch.mean((filtered_eval_y - filtered_gs_y) ** 2)).item()
                    print(f"Sample {num_loops} RMSE (filtered, {non_zero_mask.sum().item()} nodes): {sample_rmse:.6f}")
                    sample_rmses.append(sample_rmse)  # Store for later analysis
                else:
                    print(f"Sample {num_loops}: No non-zero nodes found")
                    sample_rmses.append(0.0)  # or np.nan


        num_loops += 1


    # Convert to numpy array for analysis
    sample_rmses = np.array(sample_rmses)
    
    # Concatenate all predictions and ground truths
    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)

    # Filter out padded nodes from overall RMSE calculation
    threshold = 1e-6
    non_zero_mask = np.abs(all_gts) > threshold
    filtered_all_preds = all_preds[non_zero_mask]
    filtered_all_gts = all_gts[non_zero_mask]

    # Compute overall RMSE on filtered data
    # Compute overall RMSE on filtered data
    overall_rmse = np.sqrt(np.mean((filtered_all_preds - filtered_all_gts) ** 2))
    print(f"Overall RMSE across all samples (filtered, {non_zero_mask.sum()} nodes): {overall_rmse:.6f}")
    print(f"Removed {len(all_gts) - non_zero_mask.sum()} padded/near-zero nodes from RMSE calculation")

    # ========================================================================
    # PER-SAMPLE RMSE ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PER-SAMPLE RMSE ANALYSIS")
    print("="*80)

    # Find samples with highest RMSE
    if len(sample_rmses) > 0:
        sorted_indices = np.argsort(sample_rmses)[::-1]  # Descending order
        
        print(f"\nTop 10 highest RMSE samples:")
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            print(f"  Sample {idx}: RMSE = {sample_rmses[idx]:.6f}")
        
        print(f"\nTop 10 lowest RMSE samples:")
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[-(i+1)]
            print(f"  Sample {idx}: RMSE = {sample_rmses[idx]:.6f}")
        
        print(f"\nRMSE Statistics:")
        print(f"  Mean: {np.mean(sample_rmses):.6f}")
        print(f"  Median: {np.median(sample_rmses):.6f}")
        print(f"  Std: {np.std(sample_rmses):.6f}")
        print(f"  Min: {np.min(sample_rmses):.6f}")
        print(f"  Max: {np.max(sample_rmses):.6f}")
        
        # Create histogram
        plt.figure(figsize=(8, 5))
        plt.hist(sample_rmses, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(sample_rmses), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(sample_rmses):.6f}')
        plt.axvline(np.median(sample_rmses), color='green', linestyle='--', linewidth=2, label=f'Median = {np.median(sample_rmses):.6f}')
        plt.xlabel('RMSE')
        plt.ylabel('Frequency')
        plt.title('Distribution of Per-Sample RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.postprocess_dir, 'rmse_distribution.pdf'))
        plt.close()
        print(f"\nRMSE distribution plot saved to {args.postprocess_dir}/rmse_distribution.pdf")
    
    print("="*80 + "\n")

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
    # Save as SVG instead of PDF
    plt.savefig(PATH, format='pdf', bbox_inches='tight')
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
    
    test_loss, test_rmse, sample_rmses = test(test_loader, device, best_model, mean_vec_x, std_vec_x, 
                          mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, True, args)
    
    print(f"Final Test Loss: {test_loss:.5f}")
    print(f"Final Test RMSE: {test_rmse:.5f}")
    
    # Generate visualization on test set
    plot_name = 'test_set_final_results'
    visualize(test_loader, best_model, args.postprocess_dir, plot_name, stats_list)
    
    return test_loss, test_rmse

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

best_params = {
    'model_type': 'meshgraphnet',
    'num_layers': 6,  # Updated from trial.params to optimal value
    'batch_size': 1,  # Updated from trial.params to optimal value
    'hidden_dim': 48,  # Updated from trial.params to optimal value
    'dropout_rate': 0.253,  # Updated from trial.params to optimal value
    'attention_freq': 3,  # Updated from trial.params to optimal value
    'epochs': 3000,
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
    'checkpoint_dir': checkpoint_dir,  # Use global variable
    'postprocess_dir': postprocess_dir  # Use global variable
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
best_model = MeshGraphNet(num_node_features, num_edge_features, 
                        args.hidden_dim, num_classes, args).to(device)

# Load the saved model weights
model_path = os.path.join(args.checkpoint_dir, 'model_nl6_bs4_hd48_ep1000_wd1.0673865434763588e-05_lr0.0008237166184859179_shuff_True.pt')
best_model.load_state_dict(torch.load(model_path, map_location=device))
best_model.eval()  # Set model to evaluation mode

print(f"Loaded model from: {model_path}")

# Final evaluation on test set
final_test_loss, final_test_rmse = evaluate_final_model(test_dataset, best_model, device, stats_list, args)
print(f"Final Test Loss: {final_test_loss:.5f}")
print(f"Final Test RMSE: {final_test_rmse:.5f}")

# Plot losses (since we're only testing, we don't have training/validation losses)
test_losses = [final_test_loss.item()]
save_plots(args, [0], [0], test_losses)  # Pass empty lists for train/val losses

# Generate additional visualizations on test samples
for i in range(min(5, len(test_dataset))):  # Visualize first 5 test samples
    plot_name = f'test_sample_{i}_results'
    single_loader = DataLoader([test_dataset[i]], batch_size=1, shuffle=False)
    visualize(single_loader, best_model, args.postprocess_dir, plot_name, stats_list)

plot_epochs(
    train_loss_path="./training_loss.txt", 
    val_loss_path="./val_loss.txt", 
    output_dir=postprocess_dir,
    name="model_training_history",
    transparent=False,
    text_color='black'
)