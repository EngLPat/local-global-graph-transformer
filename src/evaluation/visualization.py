"""
Visualization utilities for FCLGA GraphTransformer.

Functions for plotting strain fields, regression plots, and training curves.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import tri as mtri
from torch_geometric.loader import DataLoader

from src.utils.data_utils import normalize, unnormalize
from config import paths as config_paths


def plot_results(data, prediction, path, name, remote_stress=0.025):
    """
    Create comprehensive visualization of strain field predictions.
    
    Generates plots for actual strain, predicted strain, nominal error, and relative error.
    Saves both individual plots and comparison figures.
    
    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data containing mesh and ground truth
    prediction : torch.Tensor
        Predicted strain values from the model
    path : str
        Directory to save output plots
    name : str
        Base name for output files
    remote_stress : float, optional
        Applied remote stress for relative error calculation (default: 0.025)
    """
    print('Generating strain fields...')
    
    # Ensure data is on CPU for matplotlib processing
    pos = data.mesh_pos.cpu().numpy()
    faces = data.cells.cpu().numpy()
    
    # Prepare the ground truth, prediction, and error data
    gs_strain = data.y[:, 0].cpu().numpy()
    pred_strain = prediction[:, 0].cpu().numpy()
    nominal_error = np.abs(pred_strain - gs_strain)

    # Print diagnostic information
    print(f"Ground truth range: {gs_strain.min():.6f} to {gs_strain.max():.6f}")
    print(f"Prediction range: {pred_strain.min():.6f} to {pred_strain.max():.6f}")
    print(f"Nominal error range: {nominal_error.min():.6f} to {nominal_error.max():.6f}")

    # Define titles with mathematical notation using LaTeX
    titles = [r'Actual Strain $\varepsilon_{xx}$', 
              r'Predicted Strain $\varepsilon_{xx}$', 
              r'Error $|\varepsilon_{xx}^{pred} - \varepsilon_{xx}^{act}|$']
    
    file_names = ['actual_strain', 'predicted_strain', 'nominal_error']
    strains = [gs_strain, pred_strain, nominal_error]
    
    # Calculate element-wise (cell-wise) values by averaging node values for each triangle
    gs_element_values = np.mean(gs_strain[faces], axis=1)
    pred_element_values = np.mean(pred_strain[faces], axis=1)
    
    min_strain = min(gs_element_values.min(), pred_element_values.min())
    max_strain = max(gs_element_values.max(), pred_element_values.max())
    
    print(f"Element-wise ground truth range: {gs_element_values.min():.6f} to {gs_element_values.max():.6f}")
    print(f"Element-wise prediction range: {pred_element_values.min():.6f} to {pred_element_values.max():.6f}")
    
    # Color limits for strain fields (3: ground truth, prediction, error)
    clim = [(min_strain, max_strain), (min_strain, max_strain), (0.0001, 0.0069)]
    colormaps = ['viridis', 'viridis', 'Reds']

    # Create necessary directories
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save numerical data for later use
    data_path = os.path.join(path, name + '_data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # Save arrays for later analysis
    np.save(os.path.join(data_path, 'mesh_positions.npy'), pos)
    np.save(os.path.join(data_path, 'mesh_faces.npy'), faces)
    np.save(os.path.join(data_path, 'ground_truth.npy'), gs_strain)
    np.save(os.path.join(data_path, 'prediction.npy'), pred_strain)
    np.save(os.path.join(data_path, 'nominal_error.npy'), nominal_error)
    
    # Save as CSV format
    np.savetxt(os.path.join(data_path, 'results.csv'), 
               np.column_stack((pos, gs_strain, pred_strain, nominal_error)),
               delimiter=',', 
               header='x,y,ground_truth,prediction,nominal_error')
    
    # Create combined 3-subplot figure with updated styling
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern', 'serif'],
        'font.size': 11,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'svg.fonttype': 'none'
    })
    
    # Create figure with 3 subplots (ground truth, prediction, error with white-red colormap)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    fig.patch.set_alpha(0)
    
    for i, (ax, strain, title, clims, cmap) in enumerate(zip(axes, strains, titles, clim, colormaps)):
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=10, weight='bold')
        
        # Create the plot
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        mesh_plot = ax.tripcolor(triang, strain, shading='flat', cmap=cmap, vmin=clims[0], vmax=clims[1])
        ax.triplot(triang, 'ko-', ms=0.09, lw=0.21)
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
        
        clb.locator = ticker.MaxNLocator(nbins=5)
        clb.update_ticks()
        clb.ax.yaxis.set_tick_params(color='black')
        plt.setp(plt.getp(clb.ax.axes, 'yticklabels'), color='black')
        clb.outline.set_edgecolor('black')
    
    plt.tight_layout()
    combined_path = os.path.join(data_path, f"{name}.pdf")
    plt.savefig(combined_path, format='pdf', bbox_inches='tight', transparent=False)
    plt.close(fig)
    print(f"Saved test set prediction comparison: {combined_path}")
    
    plt.rcParams.update(plt.rcParamsDefault)
    plot_regression(gs_strain, pred_strain, data_path, name, transparent=False, text_color='black')

def plot_regression(actual_strain, predicted_strain, path, name, transparent=False, text_color='black'):
    """
    Create regression plot comparing actual vs predicted strain values.
    
    Uses hexbin visualization and filters out padded/zero nodes for cleaner results.
    
    Parameters
    ----------
    actual_strain : np.ndarray
        Ground truth strain values
    predicted_strain : np.ndarray
        Predicted strain values
    path : str
        Directory to save output plot
    name : str
        Base name for output file
    transparent : bool, optional
        Whether to use transparent background (default: False)
    text_color : str, optional
        Color for text and labels (default: 'black')
        
    Returns
    -------
    tuple
        (r_squared, rmse, num_filtered_points)
    """
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern', 'serif'],
        'font.size': 11,
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'svg.fonttype': 'none'
    })
    
    # Filter out padded nodes
    threshold = 1e-6
    non_zero_mask = np.abs(actual_strain) > threshold
    filtered_actual = actual_strain[non_zero_mask]
    filtered_predicted = predicted_strain[non_zero_mask]
    
    print(f"Original data points: {len(actual_strain)}")
    print(f"Filtered data points (actual strain > {threshold}): {len(filtered_actual)}")
    print(f"Removed {len(actual_strain) - len(filtered_actual)} padded/near-zero nodes")
    
    if len(filtered_actual) == 0:
        print("Warning: No non-zero actual strain values found. Skipping regression plot.")
        return None
    
    # Calculate metrics
    ss_res = np.sum((filtered_actual - filtered_predicted) ** 2)
    ss_tot = np.sum((filtered_actual - np.mean(filtered_actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((filtered_actual - filtered_predicted) ** 2))
    
    print(f"R-squared (R²): {r_squared:.4f}")
    print(f"RMSE: {rmse:.6f}")
    
    # Create regression plot
    fig_reg = plt.figure(figsize=(3.5, 3), dpi=300)
    fig_reg.patch.set_alpha(0)

    ax_reg = fig_reg.add_subplot(111)
    ax_reg.set_facecolor('#D0D0D0')
    ax_reg.patch.set_alpha(1.0)
    
    # Hexbin plot
    hb = ax_reg.hexbin(
        filtered_actual.flatten(),
        filtered_predicted.flatten(),
        gridsize=35, cmap='Reds', mincnt=2, vmax=50
    )
    
    # Ideal prediction line
    actual_min = np.min(filtered_actual)
    actual_max = np.max(filtered_actual)
    ax_reg.plot(
        [actual_min, actual_max],
        [actual_min, actual_max],
        color=text_color, linestyle='--', linewidth=1.0, alpha=0.7, label="Ideal prediction"
    )
    
    # Set limits with margin
    margin = (actual_max - actual_min) * 0.05
    ax_reg.set_xlim(left=max(0, actual_min - margin), right=actual_max + margin)
    ax_reg.set_ylim(bottom=max(0, actual_min - margin), top=actual_max + margin)
    
    ax_reg.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax_reg.legend(loc="upper left")
    ax_reg.grid(False)
    
    ax_reg.set_xlabel(r"Actual Strain $\varepsilon_{xx}$", color=text_color)
    ax_reg.set_ylabel(r"Predicted Strain $\varepsilon_{xx}^{\tiny\mathrm{FEA}}$", color=text_color)
    
    # Add metrics text box
    textstr = f'$R^2$ = {r_squared:.4f}\nRMSE = {rmse:.6f}'
    props = dict(boxstyle='round', facecolor=(1,1,1,0.5) if transparent else 'white', 
                 alpha=0.8, edgecolor=text_color)
    ax_reg.text(0.95, 0.05, textstr, transform=ax_reg.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right', bbox=props, color=text_color)
    
    # Style colorbar
    cbar = fig_reg.colorbar(hb, ax=ax_reg)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    cbar.outline.set_edgecolor(text_color)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)
    cbar.set_label('Counts', color=text_color, size=11)
    
    plt.tight_layout()
    reg_path = os.path.join(path, name + '_regression.pdf')
    plt.savefig(reg_path, format='pdf', bbox_inches='tight', transparent=False)
    plt.close(fig_reg)
    print(f"Regression plot saved to {reg_path}")
    
    # Save metrics to file (in same directory as regression plot)
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
    plt.rcParams.update(plt.rcParamsDefault)
    
    return r_squared, rmse, len(filtered_actual)

def load_loss_data(file_path):
    """
    Load training/validation loss data from text file.
    
    Expects format: epoch loss_value (or just loss_value per line)
    
    Parameters
    ----------
    file_path : str
        Path to loss data file
        
    Returns
    -------
    tuple
        (epochs, losses) as lists
    """
    epochs = []
    losses = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    values = line.strip().split()
                    if len(values) >= 2:
                        epochs.append(float(values[0]))
                        losses.append(float(values[1]))
                    else:
                        losses.append(float(values[0]))
                        epochs.append(len(epochs))
                        
        return epochs, losses
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [], []

def plot_epochs(train_loss_path, val_loss_path, output_dir, name="training_validation_loss", 
                transparent=False, text_color='black'):
    """
    Create plot of training and validation losses over epochs.
    
    Parameters
    ----------
    train_loss_path : str
        Path to training loss file
    val_loss_path : str
        Path to validation loss file
    output_dir : str
        Directory to save output plot
    name : str, optional
        Base name for output file (default: "training_validation_loss")
    transparent : bool, optional
        Whether to use transparent background (default: False)
    text_color : str, optional
        Color for text and labels (default: 'black')
        
    Returns
    -------
    tuple
        (training_epochs, training_losses, validation_epochs, validation_losses)
    """
    training_epochs, training_losses = load_loss_data(train_loss_path)
    validation_epochs, validation_losses = load_loss_data(val_loss_path)

    if not training_losses or not validation_losses:
        print("Training or validation loss file is missing or empty. Skipping loss plot.")
        return [], [], [], []
    
    print(f"Training: {len(training_losses)} epochs, min: {min(training_losses):.6f}, max: {max(training_losses):.6f}")
    print(f"Validation: {len(validation_losses)} epochs, min: {min(validation_losses):.6f}, max: {max(validation_losses):.6f}")
    
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern', 'serif'],
        'font.size': 11,
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'svg.fonttype': 'none'
    })
    
    fig = plt.figure(figsize=(5, 3), dpi=300)
    if transparent:
        fig.patch.set_alpha(0)
    
    ax = plt.subplot(111)
    ax.plot(training_epochs, training_losses, label="Training Loss", color="blue", linewidth=2)
    ax.plot(validation_epochs, validation_losses, label="Validation Loss", 
            linestyle="--", color="darkorange", linewidth=2)
    
    ax.set_xlabel("Epochs", fontsize=11, color=text_color)
    ax.set_ylabel("Loss", fontsize=11, color=text_color)
    
    # Set limits with margin
    all_losses = training_losses + validation_losses
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    margin = (max_loss - min_loss) * 0.1
    
    max_epoch = max(max(training_epochs), max(validation_epochs))
    ax.set_xlim(0, max_epoch + max_epoch * 0.05)
    ax.set_ylim(max(0, min_loss - margin), max_loss + margin)
    
    legend = ax.legend(loc="upper right", framealpha=0.7 if transparent else 1.0)
    for text in legend.get_texts():
        text.set_color(text_color)
    
    ax.grid(True, color=text_color, alpha=0.3, linestyle='--')
    ax.tick_params(colors=text_color, which='both')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_path = os.path.join(output_dir, f"{name}.pdf")
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', transparent=transparent)
    plt.close(fig)
    print(f"Loss plot saved to {plot_path}")
    
    return training_epochs, training_losses, validation_epochs, validation_losses

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

def save_plots(args, train_losses, val_losses, test_losses=None, velo_val_losses=None, postprocess_dir=None):
    model_name = 'model_nl' + str(args.num_layers) + '_bs' + str(args.batch_size) + \
                 '_hd' + str(args.hidden_dim) + '_ep' + str(args.epochs) + '_wd' + str(args.weight_decay) + \
                 '_lr' + str(args.lr) + '_shuff_' + str(args.shuffle) + '_tr' + str(args.train_size) + '_te' + str(args.test_size)

    if postprocess_dir is None:
        postprocess_dir = os.path.join(os.getcwd(), 'results', 'training_results')
    
    if not os.path.isdir(postprocess_dir):
        os.mkdir(postprocess_dir)

    PATH = os.path.join(postprocess_dir, 'Losses_' + model_name + '.pdf')

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


__all__ = [
    'plot_results',
    'plot_regression',
    'plot_epochs',
    'load_loss_data',
    'save_plots',
    'create_journal_quality_timing_plots',
]
