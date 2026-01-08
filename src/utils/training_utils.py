"""Training utilities for FCLGA GraphTransformer.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import os
from datetime import datetime
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.utils.data_utils import unnormalize


def create_results_folder(material_type="nonlinear"):
    """Create timestamped results folder for training run.

    Args:
        material_type: Either "linear" or "nonlinear"

    Returns:
        Path object for the created folder
    """
    # Get current timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the folder name with material type and timestamp
    result_dir = f"training_{material_type}_{timestamp}"

    # Create the full path in results/{material_type}/
    full_path = Path(os.getcwd()) / "results" / material_type / result_dir

    # Create the directory if it doesn't exist
    os.makedirs(full_path, exist_ok=True)

    return full_path


def test(
    loader,
    device,
    test_model,
    mean_vec_x,
    std_vec_x,
    mean_vec_edge,
    std_vec_edge,
    mean_vec_y,
    std_vec_y,
    is_validation,
    delta_t=0.01,
    save_model_preds=False,
    model_type=None,
):
    """
    Calculates test set losses and validation set errors.
    """
    loss = 0
    velo_rmse = 0
    num_loops = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            # Calculate the loss for the model given the test set
            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss += test_model.loss(pred, data, mean_vec_y, std_vec_y)

            # Calculate validation error if asked to
            if is_validation:
                # Unnormalize the predictions and ground truth
                eval_y = unnormalize(pred, mean_vec_y, std_vec_y)
                gs_y = data.y

                # Calculate the error
                error = torch.sum((eval_y - gs_y) ** 2, axis=1)
                velo_rmse += torch.sqrt(torch.mean(error))

        num_loops += 1

    return loss / num_loops, velo_rmse / num_loops


def evaluate_final_model(test_dataset, best_model, device, stats_list, args, postprocess_dir=None):
    """
    Evaluate the final model on the test set after training is complete.
    """
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(device),
        std_vec_x.to(device),
        mean_vec_edge.to(device),
        std_vec_edge.to(device),
        mean_vec_y.to(device),
        std_vec_y.to(device),
    )

    test_loss, test_rmse = test(
        test_loader,
        device,
        best_model,
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
        True,
    )

    # Generate visualization on test set if postprocess_dir and visualize function available
    if postprocess_dir is not None:
        try:
            from src.utils.visualization import visualize

            plot_name = "test_set_prediction_comparison"
            visualize(test_loader, best_model, postprocess_dir, plot_name, stats_list)
        except ImportError:
            print("⚠ Could not import visualize function - skipping visualization")

    return test_loss, test_rmse
