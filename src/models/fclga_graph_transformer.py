"""
FCLGA GraphTransformer Model.

Frequency-Controlled Layered Global Attention GraphTransformer for predicting
strain fields in composite materials using graph neural networks.

This model combines local message passing with periodic global attention to
efficiently capture both local and global dependencies in mesh-based simulations.

Architecture inspired by MeshGraphNets (Pfaff et al., 2021), extended with hybrid local-global attention mechanisms for improved long-range interactions. [Pfaff et al. (2021): Learning Mesh-Based Simulation with Graph Neural Networks. ICLR 2021. https://arxiv.org/abs/2010.03409]

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Linear, PReLU, Sequential

from .attention import GlobalAttention
from .processor_layer import ProcessorLayer


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
    normalized = (to_normalize - mean_vec) / std_vec
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


class FCLGA_GraphTransformer(torch.nn.Module):
    """
    Frequency-Controlled Layered Global Attention GraphTransformer.

    This model consists of three main components:
    1. Encoder: Converts raw node and edge features into latent embeddings
    2. Processor: Iterative message passing with periodic global attention
    3. Decoder: Converts latent node embeddings to output predictions

    The key innovation is the frequency-controlled global attention mechanism,
    which applies global attention every N layers to balance computational
    efficiency with global information exchange.

    Args:
        input_dim_node (int): Dimension of input node features.
        input_dim_edge (int): Dimension of input edge features.
        hidden_dim (int): Dimension of latent embeddings.
        output_dim (int): Dimension of output predictions.
        args: Configuration object with model hyperparameters including:
            - num_layers: Number of message passing layers
            - dropout_rate: Dropout probability
            - attention_freq: Apply global attention every N layers
        emb (bool, optional): Legacy parameter, not currently used.
    """

    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, args, emb=False):
        super(FCLGA_GraphTransformer, self).__init__()
        """
        FCLGA_GraphTransformer model. This model is built upon the encode-process-decode
        architecture with frequency-controlled global attention.
        
        Input_dim: dynamic variables + node_type + node_position
        Hidden_dim: 128 in original paper
        Output_dim: dynamic variables (strain predictions)
        """
        self.use_attention = True
        self.num_layers = args.num_layers
        self.dropout_rate = getattr(
            args, "dropout_rate", 0.38
        )  # Use args.dropout_rate if available

        self.attention_freq = getattr(
            args, "attention_freq", max(1, self.num_layers // 2) if self.num_layers <= 8 else 8
        )

        self.node_encoder = Sequential(
            Linear(input_dim_node, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )

        self.edge_encoder = Sequential(
            Linear(input_dim_edge, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )

        self.processor = nn.ModuleList()
        assert self.num_layers >= 1, "Number of message passing layers is not >=1"

        self.global_attention = GlobalAttention(hidden_dim)

        processor_layer = self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim, hidden_dim))

        # decoder: only for node embeddings
        self.decoder = Sequential(
            Linear(hidden_dim, hidden_dim * 2),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim * 2, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, output_dim),
        )

    def build_processor_model(self):
        """
        Factory method to build processor layers.

        Returns:
            ProcessorLayer: The processor layer class.
        """
        return ProcessorLayer

    def forward(self, data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge):
        """
        Forward pass through the FCLGA_GraphTransformer.

        The forward pass consists of three stages:
        1. Encode: Convert input features to latent embeddings
        2. Process: Iterative message passing with periodic global attention
        3. Decode: Convert latent embeddings to output predictions

        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features
                - edge_index: Edge connectivity
                - edge_attr: Edge features
            mean_vec_x (torch.Tensor): Mean for node feature normalization.
            std_vec_x (torch.Tensor): Std for node feature normalization.
            mean_vec_edge (torch.Tensor): Mean for edge feature normalization.
            std_vec_edge (torch.Tensor): Std for edge feature normalization.

        Returns:
            torch.Tensor: Predicted output values for each node.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = normalize(x, mean_vec_x, std_vec_x)
        edge_attr = normalize(edge_attr, mean_vec_edge, std_vec_edge)

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.num_layers):
            if i % self.attention_freq == 0:
                batch = getattr(data, "batch", None)
                if batch is None:
                    batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

                global_info = self.global_attention(x, batch=batch)
                x = x + 0.2 * global_info

            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        x = self.decoder(x)
        return x

    def loss(self, pred, inputs, mean_vec_y, std_vec_y):
        """
        Calculate the loss for training.

        Uses Root Mean Squared Error (RMSE) loss on normalized predictions,
        excluding fixed boundary nodes and padding nodes.

        Args:
            pred (torch.Tensor): Model predictions (normalized).
            inputs: PyTorch Geometric Data object containing ground truth.
            mean_vec_y (torch.Tensor): Mean for output normalization.
            std_vec_y (torch.Tensor): Std for output normalization.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # CRITICAL: Mask out padded nodes
        # Padded nodes have all features close to zero (especially positions)
        padding_mask = inputs.x[:, :2].abs().sum(dim=1) > 0.1  # Nodes with non-zero positions

        # Define which nodes to calculate loss for (not fixed nodes AND not padding)
        # We'll calculate loss for nodes that are not fixed boundaries and not padding
        loss_mask = (inputs.x[:, 3] < 0.5) & padding_mask  # is_fixed = 0 AND not padding

        # Normalize labels with dataset statistics
        labels = normalize(inputs.y, mean_vec_y, std_vec_y)

        # Ensure the shapes match
        if labels.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: labels shape {labels.shape} and pred shape {pred.shape} must match"
            )

        # Find sum of square errors
        error = torch.sum((labels - pred) ** 2, axis=1)

        # Root and mean the errors for the nodes we calculate loss for
        loss = torch.sqrt(torch.mean(error[loss_mask]))

        return loss
