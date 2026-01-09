"""
*******************************************************************************
*                                                                             *
*   AUTHOR: Luca Patrignani - PhD candidate Imperial College London           *
*   TITLE: FCLGA GraphTransformer Training and Testing                        *
*                                                                             *
*******************************************************************************
*                                                                             *
*  Description:                                                               *
*  ============                                                               *
*  This Python script was meticulously crafted to design and develop a GNN    *
*  to solve a mesh graph problem using PyTorch and PyTorch Geometric.         *
*  This version implements the FCLGA GraphTransformer model with advanced     *
*  attention mechanisms and message passing for structural mechanics.          *
*                                                                             *
*  Rights:                                                                    *
*  ======                                                                     *
*  All rights to this code are reserved.                                      *
*                                                                             *
*******************************************************************************
"""

import argparse
import copy

# import h5py  # Commented out - not used
# import tensorflow.compat.v1 as tf  # Commented out - not used
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import trange

# Import model components
from src.models import FCLGA_GraphTransformer

# Import utilities
from src.utils.training_utils import create_results_folder, evaluate_final_model, test

# Import visualization functions
from src.utils.visualization import save_plots

# Module-level initialization moved to main block
# print(dataset)  # Moved to main
# len(dataset_full_timesteps)/5  # Moved to main


def run_optuna_optimization(args):
    """Run Optuna hyperparameter optimization - matches legacy implementation."""
    import pickle

    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances

    # LEGACY: Set up global directories for train() function (same as legacy module-level setup)
    global checkpoint_dir, postprocess_dir
    results_folder = create_results_folder()
    checkpoint_dir = os.path.join(results_folder, "best_models")
    postprocess_dir = os.path.join(results_folder, "training_results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(postprocess_dir, exist_ok=True)

    print("=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Material type: {args.material_type}")
    print(f"Trials: {args.optuna_trials}")
    print(f"Epochs per trial: {args.epochs}")
    print("=" * 80)

    # Load dataset once for all trials
    file_path = os.path.join(
        os.getcwd(), "data", "processed", args.material_type, "datasets", "processed_data.pt"
    )

    def objective(trial):
        # Set seeds for reproducibility within each trial (LEGACY LOGIC)
        torch.manual_seed(42 + trial.number)  # Different seed per trial
        random.seed(42 + trial.number)
        np.random.seed(42 + trial.number)

        num_layers = trial.suggest_int("num_layers", 4, 8)
        attention_freq = trial.suggest_int("attention_freq", 2, num_layers)  # Always optimize

        trial_args = {
            "model_type": "fclga",
            "num_layers": num_layers,
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 12]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [48, 64, 96, 128]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.3),
            "attention_freq": attention_freq,
            "epochs": args.epochs,
            "opt": trial.suggest_categorical("opt", ["adam", "rmsprop"]),
            "opt_scheduler": "step",
            "opt_decay_step": trial.suggest_int("opt_decay_step", 40, 80),
            "opt_decay_rate": trial.suggest_float("opt_decay_rate", 0.65, 0.85),
            "opt_restart": 0,
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 8e-3, log=True),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "shuffle": True,
            "save_velo_val": True,
            "save_best_model": False,  # Don't save during optimization
        }

        # LEGACY: Memory constraint
        if trial_args["hidden_dim"] > 64 and trial_args["batch_size"] > 8:
            trial_args["num_layers"] = min(trial_args["num_layers"], 6)

        # Convert to objectview
        trial_args_obj = objectview(trial_args)

        try:
            # Load and split dataset using configured ratios
            dataset = torch.load(file_path, weights_only=False)

            # Calculate split sizes from ratios
            total_size = len(dataset)
            train_size = int(total_size * trial_args.get("train_ratio", 0.7))
            val_size = int(total_size * trial_args.get("val_ratio", 0.15))

            # Create the splits
            if trial_args["shuffle"]:
                random.shuffle(dataset)

            train_dataset = dataset[:train_size]
            val_dataset = dataset[train_size : train_size + val_size]

            # Update args
            trial_args_obj.train_size = train_size
            trial_args_obj.val_size = val_size
            trial_args_obj.test_size = total_size - train_size - val_size

            # LEGACY: Get statistics for normalization (only use training data)
            stats_list = get_stats(train_dataset)

            # Train model
            print(
                f"\nTrial {trial.number}: layers={trial_args['num_layers']}, "
                f"hidden={trial_args['hidden_dim']}, lr={trial_args['lr']:.2e}"
            )

            val_losses, losses, _, _ = train(
                train_dataset, val_dataset, trial_args_obj.device, stats_list, trial_args_obj
            )

            # Clean up GPU memory
            torch.cuda.empty_cache()

            # Return best validation loss
            return min(val_losses)

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Return a large value to indicate failure
            return float("inf")

    # Create study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.optuna_trials)

    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Create organized results directory inside training run folder
    opt_results_dir = Path(results_folder) / "hyperparameter_optimization_results"
    opt_results_dir.mkdir(exist_ok=True)

    # Save study pickle file
    study_path = opt_results_dir / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"\n✓ Study saved to {study_path}")

    # Save best hyperparameters as readable text file
    best_params_path = opt_results_dir / "best_hyperparameters.txt"
    with open(best_params_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of trials: {args.optuna_trials}\n")
        f.write(f"Epochs per trial: {args.epochs}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best validation loss: {study.best_value:.6f}\n\n")
        f.write("Best hyperparameters:\n")
        f.write("-" * 40 + "\n")
        for key, value in study.best_params.items():
            f.write(f"  {key:20s} : {value}\n")
        f.write("\n" + "=" * 80 + "\n")
    print(f"✓ Best hyperparameters saved to {best_params_path}")

    # Save best hyperparameters as JSON for automated loading by test script
    best_params_json = opt_results_dir / "best_hyperparameters.json"
    with open(best_params_json, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"✓ Machine-readable hyperparameters saved to {best_params_json}")

    # Generate visualization plots
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image(str(opt_results_dir / "optimization_history.png"))

        fig2 = plot_param_importances(study)
        fig2.write_image(str(opt_results_dir / "param_importances.png"))

        print(f"✓ Visualizations saved to {opt_results_dir}/")
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")

    print("=" * 80)

    # LEGACY: Train final model with best hyperparameters if requested
    if args.final_epochs is not None:
        print("\n" + "=" * 80)
        print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
        print("=" * 80)
        print(f"Epochs: {args.final_epochs}")

        # LEGACY: Start with all trial params (like create_best_params_from_trial)
        best_params = study.best_params.copy()

        # LEGACY: Reconstruct attention_freq if not in params (happens when num_layers <= 4)
        if "attention_freq" not in best_params:
            num_layers = best_params["num_layers"]
            if num_layers <= 4:
                best_params["attention_freq"] = num_layers
            elif num_layers <= 6:
                # Should not happen, but use num_layers as fallback
                best_params["attention_freq"] = num_layers
            else:
                # Should not happen, but use 4 as fallback
                best_params["attention_freq"] = 4

        # Override/add specific parameters for final training
        best_params.update(
            {
                "model_type": "fclga",
                "epochs": args.final_epochs,
                "opt_scheduler": "step",
                "opt_restart": 0,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "shuffle": True,
                "save_velo_val": True,
                "save_best_model": True,
            }
        )

        best_args = objectview(best_params)

        # Load dataset and create splits using configured ratios
        dataset = torch.load(file_path, weights_only=False)
        total_size = len(dataset)
        train_size = int(total_size * best_args.train_ratio)
        val_size = int(total_size * best_args.val_ratio)
        test_size = total_size - train_size - val_size

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        if best_args.shuffle:
            random.shuffle(dataset)

        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size : train_size + val_size]
        test_dataset = dataset[train_size + val_size :]

        # Update args with actual sizes
        best_args.train_size = train_size
        best_args.val_size = val_size
        best_args.test_size = test_size

        # Get statistics for normalization (only from training data)
        stats_list = get_stats(train_dataset)

        # Train final model
        print("\nTraining with best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"  epochs: {args.final_epochs}")
        print("=" * 80 + "\n")

        val_losses, losses, velo_val_losses, best_model = train(
            train_dataset, val_dataset, best_args.device, stats_list, best_args
        )

        # Evaluate on test set
        print("\nEvaluating final model on test set...")
        final_test_loss, final_test_rmse = evaluate_final_model(
            test_dataset, best_model, best_args.device, stats_list, best_args, postprocess_dir
        )

        print(f"\nFinal Test Loss: {final_test_loss:.5f}")
        print(f"Final Test RMSE: {final_test_rmse:.5f}")

        # Save final best model
        model_name = (
            "model_nl"
            + str(best_args.num_layers)
            + "_bs"
            + str(best_args.batch_size)
            + "_hd"
            + str(best_args.hidden_dim)
            + "_ep"
            + str(best_args.epochs)
            + "_wd"
            + str(best_args.weight_decay)
            + "_lr"
            + str(best_args.lr)
            + "_shuff_"
            + str(best_args.shuffle)
            + "_tr"
            + str(best_args.train_size)
            + "_te"
            + str(best_args.test_size)
        )
        model_path = os.path.join(checkpoint_dir, model_name + "_FINAL.pt")
        torch.save(best_model.state_dict(), model_path)
        print(f"\n✓ Final model saved to: {model_path}")

        # Save final plots
        test_losses = [final_test_loss.item()]
        save_plots(best_args, losses, val_losses, test_losses, postprocess_dir=postprocess_dir)

        print("=" * 80)
        print("FINAL MODEL TRAINING COMPLETE")
        print(f"Results saved in: {results_folder}")
        print(f"  - Model: {checkpoint_dir}")
        print(f"  - Training results: {postprocess_dir}")
        print("=" * 80)


def normalize(to_normalize, mean_vec, std_vec):
    normalized = (to_normalize - mean_vec) / std_vec
    return normalized


def unnormalize(to_unnormalize, mean_vec, std_vec):
    return to_unnormalize * std_vec + mean_vec


def get_stats(data_list):
    """
    Method for normalizing processed datasets. Given  the processed data_list,
    calculates the mean and standard deviation for the node features, edge features,
    and node outputs, and normalizes these using the calculated statistics.
    """

    # mean and std of the node features are calculated
    mean_vec_x = torch.zeros(data_list[0].x.shape[1:])
    std_vec_x = torch.zeros(data_list[0].x.shape[1:])

    # mean and std of the edge features are calculated
    mean_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])

    # mean and std of the output parameters are calculated
    mean_vec_y = torch.zeros(data_list[0].y.shape[1:])
    std_vec_y = torch.zeros(data_list[0].y.shape[1:])

    # Define the maximum number of accumulations to perform such that we do
    # not encounter memory issues
    max_accumulations = 10**6

    # Define a very small value for normalizing to
    eps = torch.tensor(1e-8)

    # Define counters used in normalization
    num_accs_x = 0
    num_accs_edge = 0
    num_accs_y = 0

    # Iterate through the data in the list to accumulate statistics
    for dp in data_list:
        # Add to the
        mean_vec_x += torch.sum(dp.x, dim=0)
        std_vec_x += torch.sum(dp.x**2, dim=0)
        num_accs_x += dp.x.shape[0]

        mean_vec_edge += torch.sum(dp.edge_attr, dim=0)
        std_vec_edge += torch.sum(dp.edge_attr**2, dim=0)
        num_accs_edge += dp.edge_attr.shape[0]

        mean_vec_y += torch.sum(dp.y, dim=0)
        std_vec_y += torch.sum(dp.y**2, dim=0)
        num_accs_y += dp.y.shape[0]

        if (
            num_accs_x > max_accumulations
            or num_accs_edge > max_accumulations
            or num_accs_y > max_accumulations
        ):
            break

    mean_vec_x = mean_vec_x / num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x / num_accs_x - mean_vec_x**2), eps)

    mean_vec_edge = mean_vec_edge / num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge / num_accs_edge - mean_vec_edge**2), eps)

    mean_vec_y = mean_vec_y / num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y / num_accs_y - mean_vec_y**2), eps)

    mean_std_list = [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y]

    return mean_std_list


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == "adam":
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == "none":
        return None, optimizer
    elif args.opt_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate
        )
    elif args.opt_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    elif args.opt_scheduler == "cosine":
        # Cosine annealing with warmup
        def lr_lambda(current_step: int):
            warmup_steps = 200
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, args.epochs - warmup_steps)
            )
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
    print(
        f"  Average displacement (for displaced nodes): {torch.sum(displacement_amount * is_displaced) / torch.sum(is_displaced)}"
    )

    print("=================================\n")


def train(train_dataset, val_dataset, device, stats_list, args):
    """
    Performs a training loop on the dataset for FCLGA GraphTransformer with proper validation.
    """
    df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "velo_val_loss"])

    # Define the model name for saving
    model_name = (
        "model_nl"
        + str(args.num_layers)
        + "_bs"
        + str(args.batch_size)
        + "_hd"
        + str(args.hidden_dim)
        + "_ep"
        + str(args.epochs)
        + "_wd"
        + str(args.weight_decay)
        + "_lr"
        + str(args.lr)
        + "_shuff_"
        + str(args.shuffle)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # The statistics of the data are decomposed
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(device),
        std_vec_x.to(device),
        mean_vec_edge.to(device),
        std_vec_edge.to(device),
        mean_vec_y.to(device),
        std_vec_y.to(device),
    )

    # Build model
    num_node_features = train_dataset[0].x.shape[1]
    num_edge_features = train_dataset[0].edge_attr.shape[1]
    num_classes = 1

    model = FCLGA_GraphTransformer(
        num_node_features, num_edge_features, args.hidden_dim, num_classes, args
    ).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # Train
    losses = []
    val_losses = []
    velo_val_losses = []
    best_val_loss = float("inf")
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
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # Prevent gradient explosion
            opt.step()
            total_loss += loss.item()
            num_loops += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / num_loops
        losses.append(train_loss)
        # Validation step
        model.eval()
        val_loss, velo_val_rmse = test(
            val_loader,
            device,
            model,
            mean_vec_x,
            std_vec_x,
            mean_vec_edge,
            std_vec_edge,
            mean_vec_y,
            std_vec_y,
            args.save_velo_val,
        )

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
        new_row = pd.DataFrame(
            {"epoch": [epoch], "train_loss": [train_loss], "val_loss": [val_loss.item()]}
        )
        if args.save_velo_val:
            new_row["velo_val_loss"] = velo_val_rmse.item()
        df = pd.concat([df, new_row], ignore_index=True)

        if args.save_best_model and epoch % 5 == 0:
            PATH = os.path.join(checkpoint_dir, model_name + ".pt")
            torch.save(best_model.state_dict(), PATH)

    # Save final dataframe
    PATH = os.path.join(checkpoint_dir, model_name + ".csv")
    df.to_csv(PATH, index=False)

    # Note: Visualization moved to test set evaluation only (more meaningful)

    return val_losses, losses, velo_val_losses, best_model


class objectview:
    def __init__(self, d):
        self.__dict__ = d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FCLGA GraphTransformer for structural mechanics prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default parameters
  python src/training/fclga_train_model.py --epochs 500
  
  # Custom hyperparameters
  python src/training/fclga_train_model.py --epochs 1000 --num_layers 6 --hidden_dim 64 --lr 0.001
  
  # Full specification
  python src/training/fclga_train_model.py \\
      --num_layers 6 --batch_size 4 --hidden_dim 48 --dropout_rate 0.25 \\
      --attention_freq 3 --epochs 500 --lr 0.000824 --weight_decay 1.07e-05
  
  # Using GPU
  python src/training/fclga_train_model.py --epochs 500 --device cuda

Default values are optimized for the open-hole plate problem.
For different geometries, consider adjusting hyperparameters.
        """,
    )

    # Model Architecture
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of message passing layers (default: 6)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=48, help="Hidden dimension size (default: 48)"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.253, help="Dropout rate (default: 0.253)"
    )
    parser.add_argument(
        "--attention_freq",
        type=int,
        default=3,
        help="Global attention frequency - apply every N layers (default: 3)",
    )

    # Training Parameters
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of training epochs (default: 500)"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument(
        "--lr", type=float, default=8.24e-04, help="Learning rate (default: 8.24e-04)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1.07e-05,
        help="Weight decay for optimizer (default: 1.07e-05)",
    )

    # Optimizer Settings
    parser.add_argument(
        "--opt",
        type=str,
        default="adam",
        choices=["adam", "rmsprop", "sgd"],
        help="Optimizer type (default: adam)",
    )
    parser.add_argument(
        "--opt_scheduler",
        type=str,
        default="step",
        choices=["step", "cosine", "none"],
        help="Learning rate scheduler (default: step)",
    )
    parser.add_argument(
        "--opt_decay_step", type=int, default=46, help="Decay step for step scheduler (default: 46)"
    )
    parser.add_argument(
        "--opt_decay_rate",
        type=float,
        default=0.668,
        help="Decay rate for step scheduler (default: 0.668)",
    )

    # Dataset Parameters
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Training set ratio (default: 0.7 = 70%%)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="Test set ratio (default: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=True, help="Shuffle dataset (default: True)"
    )
    parser.add_argument(
        "--no_shuffle", dest="shuffle", action="store_false", help="Do not shuffle dataset"
    )

    # Device and Output
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto - uses CUDA if available)",
    )
    parser.add_argument(
        "--material_type",
        type=str,
        default="nonlinear",
        choices=["linear", "nonlinear"],
        help="Material type: linear (elastic) or nonlinear (plastic) - default: nonlinear",
    )
    parser.add_argument(
        "--save_best_model",
        action="store_true",
        default=True,
        help="Save best model during training (default: True)",
    )
    parser.add_argument(
        "--save_velo_val",
        action="store_true",
        default=True,
        help="Save velocity validation metrics (default: True)",
    )

    # Hyperparameter Optimization
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization instead of single training",
    )
    parser.add_argument(
        "--optuna_trials",
        type=int,
        default=10,
        help="Number of Optuna optimization trials (default: 10)",
    )
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=None,
        help="After optimization, train final model with best hyperparameters for this many epochs (default: None - no final training)",
    )

    args = parser.parse_args()

    # If optimization requested, run Optuna and exit
    if args.optimize:
        try:
            import optuna
        except ImportError:
            print("ERROR: Optuna not installed. Install with: pip install optuna")
            print("Falling back to single training run...")
        else:
            run_optuna_optimization(args)
            exit(0)

    # Convert args to dict and set additional parameters
    args_dict = vars(args).copy()
    args_dict["model_type"] = "fclga"
    args_dict["opt_restart"] = 0

    # Handle device auto-selection
    if args.device == "auto":
        args_dict["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        args_dict["device"] = args.device

    # Setup directories first
    results_folder = create_results_folder()
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, "datasets")
    checkpoint_dir = os.path.join(results_folder, "best_models")
    postprocess_dir = os.path.join(results_folder, "training_results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(postprocess_dir, exist_ok=True)

    # Set directory paths in args
    args_dict["checkpoint_dir"] = checkpoint_dir
    args_dict["postprocess_dir"] = postprocess_dir

    args = objectview(args_dict)

    print("=" * 80)
    print("FCLGA GraphTransformer Training")
    print("=" * 80)
    print("Model Configuration:")
    print(f"  - Layers: {args.num_layers}")
    print(f"  - Hidden dim: {args.hidden_dim}")
    print(f"  - Attention frequency: {args.attention_freq}")
    print(f"  - Dropout: {args.dropout_rate}")
    print("\nTraining Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"  - Optimizer: {args.opt}")
    print(f"  - Device: {args.device}")
    print("\nDataset Configuration:")
    print(f"  - Training samples: {args.train_size}")
    print(f"  - Test samples: {args.test_size}")
    print(f"  - Shuffle: {args.shuffle}")
    print("=" * 80)
    print()

    # Initialize directories
    results_folder = create_results_folder(material_type=args.material_type)
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, "data", "processed", args.material_type, "datasets")
    checkpoint_dir = os.path.join(results_folder, "best_models")
    postprocess_dir = os.path.join(results_folder, "training_results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(postprocess_dir, exist_ok=True)

    # Update args with directory paths
    args.checkpoint_dir = checkpoint_dir
    args.postprocess_dir = postprocess_dir

    file_path = os.path.join(dataset_dir, "processed_data.pt")

    torch.manual_seed(5)  # Torch
    random.seed(5)  # Python
    np.random.seed(5)  # NumPy

    dataset = torch.load(file_path, weights_only=False)[: (args.train_size + args.test_size)]

    sample_data = dataset[0]  # Get first graph in dataset

    print("Node feature tensor shape:", sample_data.x.shape)
    print("Number of nodes:", sample_data.x.shape[0])
    print("Node feature dimension:", sample_data.x.shape[1])

    # Print statistics
    print(
        "Node features min/max/mean:",
        torch.min(sample_data.x).item(),
        torch.max(sample_data.x).item(),
        torch.mean(sample_data.x).item(),
    )
    print("First 3 node features:")
    for i in range(min(3, sample_data.x.shape[0])):
        print(f"Node {i}:", sample_data.x[i])

    print("\nExample node types (if one-hot encoded):")
    if sample_data.x.shape[1] > 3:  # Assuming at least a few features
        # Look at potential one-hot encoded section (often in latter part of feature vector)
        potential_onehot = sample_data.x[:5, 2:]  # First 5 nodes, features from 3rd onward
        print(potential_onehot)

        # Check if any rows sum to 1 (typical of one-hot encoding)
        row_sums = torch.sum(potential_onehot, dim=1)
        print("Sum of potential one-hot section:", row_sums)

    analyze_node_features(dataset)

    # Calculate split sizes from configured ratios
    total_size = len(dataset)
    train_size = int(total_size * args.train_ratio)
    val_size = int(total_size * args.val_ratio)
    test_size = total_size - train_size - val_size

    print(f"Dataset size: {total_size}")
    print(f"Split ratios: {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Test set size: {test_size}")

    # Create the splits
    if args.shuffle:
        random.shuffle(dataset)

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size : train_size + val_size]
    test_dataset = dataset[train_size + val_size :]

    # Update args
    args.train_size = train_size
    args.val_size = val_size
    args.test_size = test_size

    # Get statistics for normalization (use only training data to prevent data leakage)
    stats_list = get_stats(train_dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(device)

    val_losses, losses, velo_val_losses, best_model = train(
        train_dataset, val_dataset, device, stats_list, args
    )

    # Evaluate on test set
    print("\nEvaluating final model on test set...")
    final_test_loss, final_test_rmse = evaluate_final_model(
        test_dataset, best_model, device, stats_list, args, postprocess_dir
    )

    print(f"\nFinal Test Loss: {final_test_loss:.5f}")
    print(f"Final Test RMSE: {final_test_rmse:.5f}")

    # Save final best model
    model_name = (
        "model_nl"
        + str(args.num_layers)
        + "_bs"
        + str(args.batch_size)
        + "_hd"
        + str(args.hidden_dim)
        + "_ep"
        + str(args.epochs)
        + "_wd"
        + str(args.weight_decay)
        + "_lr"
        + str(args.lr)
        + "_shuff_"
        + str(args.shuffle)
        + "_tr"
        + str(args.train_size)
        + "_te"
        + str(args.test_size)
    )
    model_path = os.path.join(checkpoint_dir, model_name + "_FINAL.pt")
    torch.save(best_model.state_dict(), model_path)
    print(f"\n✓ Final model saved to: {model_path}")

    # Save final plots
    test_losses = [final_test_loss.item()]
    save_plots(args, losses, val_losses, test_losses, postprocess_dir=postprocess_dir)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Results saved in: {results_folder}")
    print(f"  - Models: {checkpoint_dir}")
    print(f"  - Training results: {postprocess_dir}")
    print("=" * 80)
