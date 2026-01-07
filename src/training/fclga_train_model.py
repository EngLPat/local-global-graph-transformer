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
# import h5py  # Commented out - not used
# import tensorflow.compat.v1 as tf  # Commented out - not used
import functools
import json
from torch_geometric.data import Data
import enum
import math
import argparse

import os
import datetime
from pathlib import Path

import time
import pandas as pd
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import numpy as np

# Import utilities
from src.utils.training_utils import (
    create_results_folder, 
    test, 
    evaluate_final_model
)
from src.utils.benchmark_utils import (
    benchmark_attention_frequencies,
    analyze_timing_results,
    create_summary_table,
    save_timing_results,
    benchmark_inference_time,
    save_speedup_analysis
)

# Import visualization functions
from src.utils.visualization import visualize, plot_results, save_plots

# Module-level initialization moved to main block
# print(dataset)  # Moved to main
# len(dataset_full_timesteps)/5  # Moved to main


def run_optuna_optimization(args):
    """Run Optuna hyperparameter optimization - matches legacy implementation."""
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances
    import pickle
    
    # LEGACY: Set up global directories for train() function (same as legacy module-level setup)
    global checkpoint_dir, postprocess_dir
    results_folder = create_results_folder()
    checkpoint_dir = os.path.join(results_folder, 'best_models')
    postprocess_dir = os.path.join(results_folder, 'plots')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(postprocess_dir, exist_ok=True)
    
    print("="*80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Trials: {args.optuna_trials}")
    print(f"Epochs per trial: {args.epochs}")
    print("="*80)
    
    # Load dataset once for all trials
    file_path = os.path.join(os.getcwd(), 'datasets', 'processed_data.pt')
    if not os.path.exists(file_path):
        file_path = os.path.join(os.getcwd(), 'data', 'processed', 'datasets', 'processed_data.pt')
    
    def objective(trial):
        # Set seeds for reproducibility within each trial (LEGACY LOGIC)
        torch.manual_seed(42 + trial.number)  # Different seed per trial
        random.seed(42 + trial.number)
        np.random.seed(42 + trial.number)
        
        num_layers = trial.suggest_int('num_layers', 4, 8)

        # LEGACY: Conditional attention frequency bounds based on num_layers
        if num_layers <= 4:
            attention_freq = num_layers  # Only at the end for very small models
        elif num_layers <= 6:
            attention_freq = trial.suggest_int('attention_freq', 3, num_layers)  # Every 3+ layers
        else:
            attention_freq = trial.suggest_int('attention_freq', 4, 8)  # Every 4-8 layers

        trial_args = {
            'model_type': 'fclga',
            'num_layers': num_layers,
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 12]),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [48, 64, 96, 128]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'attention_freq': attention_freq,  # Use the conditional logic
            'epochs': args.epochs,
            'opt': trial.suggest_categorical('opt', ['adam', 'rmsprop']),
            'opt_scheduler': 'step',
            'opt_decay_step': trial.suggest_int('opt_decay_step', 40, 80),
            'opt_decay_rate': trial.suggest_float('opt_decay_rate', 0.65, 0.85),
            'opt_restart': 0,
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True),
            'lr': trial.suggest_float('lr', 1e-4, 8e-3, log=True),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'shuffle': True,
            'save_velo_val': True,
            'save_best_model': False,  # Don't save during optimization
        }
        
        # LEGACY: Memory constraint
        if trial_args['hidden_dim'] > 64 and trial_args['batch_size'] > 8:
            trial_args['num_layers'] = min(trial_args['num_layers'], 6)
        
        # Convert to objectview
        trial_args_obj = objectview(trial_args)
        
        try:
            # LEGACY: Load and split dataset with 70/15/15 split
            dataset = torch.load(file_path, weights_only=False)
            
            # Calculate split sizes
            total_size = len(dataset)
            train_size = int(total_size * 0.7)
            val_size = int(total_size * 0.15)
            
            # Create the splits
            if trial_args['shuffle']:
                random.shuffle(dataset)
            
            train_dataset = dataset[:train_size]
            val_dataset = dataset[train_size:train_size+val_size]
            
            # Update args
            trial_args_obj.train_size = train_size
            trial_args_obj.val_size = val_size
            trial_args_obj.test_size = total_size - train_size - val_size
            
            # LEGACY: Get statistics for normalization (only use training data)
            stats_list = get_stats(train_dataset)
            
            # Train model
            print(f"\nTrial {trial.number}: layers={trial_args['num_layers']}, "
                  f"hidden={trial_args['hidden_dim']}, lr={trial_args['lr']:.2e}")
            
            val_losses, losses, _, _ = train(train_dataset, val_dataset, trial_args_obj.device, 
                                              stats_list, trial_args_obj)
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            # Return best validation loss
            return min(val_losses)
        
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Return a large value to indicate failure
            return float('inf')
    
    # Create study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.optuna_trials)
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save study
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    study_path = results_dir / 'optuna_study.pkl'
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"\n✓ Study saved to {study_path}")
    
    # Generate visualization plots
    try:
        viz_dir = results_dir / 'optuna_visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        fig1 = plot_optimization_history(study)
        fig1.write_image(str(viz_dir / 'optimization_history.png'))
        
        fig2 = plot_param_importances(study)
        fig2.write_image(str(viz_dir / 'param_importances.png'))
        
        print(f"✓ Visualizations saved to {viz_dir}/")
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")
    
    print("="*80)
    
    # LEGACY: Train final model with best hyperparameters if requested
    if args.final_epochs is not None:
        print("\n" + "="*80)
        print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
        print("="*80)
        print(f"Epochs: {args.final_epochs}")
        
        # LEGACY: Start with all trial params (like create_best_params_from_trial)
        best_params = study.best_params.copy()
        
        # LEGACY: Reconstruct attention_freq if not in params (happens when num_layers <= 4)
        if 'attention_freq' not in best_params:
            num_layers = best_params['num_layers']
            if num_layers <= 4:
                best_params['attention_freq'] = num_layers
            elif num_layers <= 6:
                # Should not happen, but use num_layers as fallback
                best_params['attention_freq'] = num_layers
            else:
                # Should not happen, but use 4 as fallback
                best_params['attention_freq'] = 4
        
        # Override/add specific parameters for final training
        best_params.update({
            'model_type': 'fclga',
            'epochs': args.final_epochs,
            'opt_scheduler': 'step',
            'opt_restart': 0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'shuffle': True,
            'save_velo_val': True,
            'save_best_model': True,
        })
        
        best_args = objectview(best_params)
        
        # Load dataset and create splits (same as optimization)
        dataset = torch.load(file_path, weights_only=False)
        total_size = len(dataset)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.15)
        test_size = total_size - train_size - val_size
        
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        
        if best_args.shuffle:
            random.shuffle(dataset)
        
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size+val_size]
        test_dataset = dataset[train_size+val_size:]
        
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
        print("="*80 + "\n")
        
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
        
        # Save final plots
        test_losses = [final_test_loss.item()]
        save_plots(best_args, losses, val_losses, test_losses, postprocess_dir=postprocess_dir)
        
        print("="*80)
        print("FINAL MODEL TRAINING COMPLETE")
        print("="*80)

def normalize(to_normalize, mean_vec, std_vec):
    normalized = (to_normalize - mean_vec) / std_vec
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
        self.hidden_dim = hidden_dim  # ADD THIS LINE - store hidden_dim as instance variable
        self.dropout_rate = getattr(args, 'dropout_rate', 0.38)  # Use args.dropout_rate if available
        self.operation_count = 0  # Add operation counter
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

        self.operation_count = 0

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension
        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # Step 2: perform message passing with latent node/edge embeddings
        layer_outputs = [x]  # Store layer outputs for skip connections
        for i in range(self.num_layers):
            # Count message passing operations
            num_edges = data.edge_index.shape[1]
            message_ops = num_edges * (self.hidden_dim ** 2)
            self.operation_count += message_ops
            # Add skip connection from 8 layers back (keep existing logic)
            if i >= 8 and i % 8 == 0:  # Every 8 layers after the 8th
                x = x + self.skip_projection(layer_outputs[i-8])
                
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)
            layer_outputs.append(x)  # Store current layer output
            
            # FIXED: Adaptive global attention frequency
            if (i + 1) % self.attention_freq == 0:
                num_nodes = x.size(0)
                attention_ops = num_nodes ** 2 * self.hidden_dim
                self.operation_count += attention_ops
                # Create batch index if not provided
                batch = getattr(data, 'batch', None)
                if batch is None:
                    batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
                    
                global_info = self.global_attention(x, batch=batch)
                x = x + 0.2 * global_info  # Mix with local representations

        # step 3: decode latent node embeddings into physical quantities of interest
        return self.decoder(x)
    
    def get_operation_count(self):
        return self.operation_count
    
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
    
    # Plot results from final model (example prediction on validation set)
    plot_name = 'validation_set_prediction_example'
    visualize(val_loader, best_model, postprocess_dir, plot_name, stats_list)
    
    return val_losses, losses, velo_val_losses, best_model

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train FCLGA GraphTransformer for structural mechanics prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
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
        '''
    )
    
    # Model Architecture
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of message passing layers (default: 6)')
    parser.add_argument('--hidden_dim', type=int, default=48,
                        help='Hidden dimension size (default: 48)')
    parser.add_argument('--dropout_rate', type=float, default=0.253,
                        help='Dropout rate (default: 0.253)')
    parser.add_argument('--attention_freq', type=int, default=3,
                        help='Global attention frequency - apply every N layers (default: 3)')
    
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=8.24e-04,
                        help='Learning rate (default: 8.24e-04)')
    parser.add_argument('--weight_decay', type=float, default=1.07e-05,
                        help='Weight decay for optimizer (default: 1.07e-05)')
    
    # Optimizer Settings
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'rmsprop', 'sgd'],
                        help='Optimizer type (default: adam)')
    parser.add_argument('--opt_scheduler', type=str, default='step', choices=['step', 'cosine', 'none'],
                        help='Learning rate scheduler (default: step)')
    parser.add_argument('--opt_decay_step', type=int, default=46,
                        help='Decay step for step scheduler (default: 46)')
    parser.add_argument('--opt_decay_rate', type=float, default=0.668,
                        help='Decay rate for step scheduler (default: 0.668)')
    
    # Dataset Parameters
    parser.add_argument('--train_size', type=int, default=400,
                        help='Number of training samples (default: 400)')
    parser.add_argument('--test_size', type=int, default=100,
                        help='Number of test samples (default: 100)')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle dataset (default: True)')
    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false',
                        help='Do not shuffle dataset')
    
    # Device and Output
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto - uses CUDA if available)')
    parser.add_argument('--save_best_model', action='store_true', default=True,
                        help='Save best model during training (default: True)')
    parser.add_argument('--save_velo_val', action='store_true', default=True,
                        help='Save velocity validation metrics (default: True)')
    
    # Hyperparameter Optimization
    parser.add_argument('--optimize', action='store_true',
                        help='Run Optuna hyperparameter optimization instead of single training')
    parser.add_argument('--optuna_trials', type=int, default=10,
                        help='Number of Optuna optimization trials (default: 10)')
    parser.add_argument('--final_epochs', type=int, default=None,
                        help='After optimization, train final model with best hyperparameters for this many epochs (default: None - no final training)')
    
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
    args_dict['model_type'] = 'fclga'
    args_dict['opt_restart'] = 0
    
    # Handle device auto-selection
    if args.device == 'auto':
        args_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        args_dict['device'] = args.device
    
    # Setup directories first
    results_folder = create_results_folder()
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, 'datasets')
    checkpoint_dir = os.path.join(results_folder, 'best_models')
    postprocess_dir = os.path.join(results_folder, 'plots')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(postprocess_dir, exist_ok=True)
    
    # Set directory paths in args
    args_dict['checkpoint_dir'] = checkpoint_dir
    args_dict['postprocess_dir'] = postprocess_dir
    
    args = objectview(args_dict)
    
    print("="*80)
    print("FCLGA GraphTransformer Training")
    print("="*80)
    print(f"Model Configuration:")
    print(f"  - Layers: {args.num_layers}")
    print(f"  - Hidden dim: {args.hidden_dim}")
    print(f"  - Attention frequency: {args.attention_freq}")
    print(f"  - Dropout: {args.dropout_rate}")
    print(f"\nTraining Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"  - Optimizer: {args.opt}")
    print(f"  - Device: {args.device}")
    print(f"\nDataset Configuration:")
    print(f"  - Training samples: {args.train_size}")
    print(f"  - Test samples: {args.test_size}")
    print(f"  - Shuffle: {args.shuffle}")
    print("="*80)
    print()

    # Initialize directories
    results_folder = create_results_folder()
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, 'datasets')
    checkpoint_dir = os.path.join(results_folder, 'best_models')
    postprocess_dir = os.path.join(results_folder, 'plots')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(postprocess_dir, exist_ok=True)
    
    # Update args with directory paths
    args.checkpoint_dir = checkpoint_dir
    args.postprocess_dir = postprocess_dir
    
    file_path = os.path.join(dataset_dir, 'processed_data.pt')

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
    
    val_losses, losses, velo_val_losses, best_model = train(
        train_dataset, val_dataset, device, stats_list, args
    )