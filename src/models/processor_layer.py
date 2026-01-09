"""
Processor Layer for FCLGA GraphTransformer.

This module implements the message passing layer that updates node and edge features
through iterative message passing and aggregation operations.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import torch
import torch.nn as nn
import torch_scatter
from torch.nn import LayerNorm, Linear, PReLU, Sequential
from torch_geometric.nn.conv import MessagePassing


class ProcessorLayer(MessagePassing):
    """
    Message passing processor layer for graph neural networks.

    This layer performs edge and node feature updates through message passing.
    Edge features are updated based on connected node features, then node features
    are updated by aggregating messages from neighboring edges.

    Args:
        in_channels (int): Dimension of input node embeddings.
        out_channels (int): Dimension of output node embeddings.
        **kwargs: Additional arguments for MessagePassing base class.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        self.attention = nn.Sequential(Linear(3 * in_channels, 1), nn.Sigmoid())

        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.edge_mlp = Sequential(
            Linear(3 * in_channels, out_channels),
            PReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )

        self.node_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            PReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        """Message passing: update edges, aggregate to nodes, apply residual connection."""
        out, updated_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=size
        )

        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = x + self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """Update edge features using source node, target node, and current edge features."""
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edges = self.edge_mlp(updated_edges) + edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size=None):
        """Aggregate edge messages to target nodes via scatter sum."""
        node_dim = 0
        out = torch_scatter.scatter(
            updated_edges, edge_index[0, :], dim=node_dim, dim_size=dim_size, reduce="sum"
        )

        return out, updated_edges
