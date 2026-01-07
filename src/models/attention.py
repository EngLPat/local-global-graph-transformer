"""
Global Attention Module for FCLGA GraphTransformer.

This module implements a global attention mechanism that allows nodes in the graph
to exchange information globally, beyond local message passing neighborhoods.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import torch
from torch.nn import Linear


class GlobalAttention(torch.nn.Module):
    """
    Global attention mechanism for graph neural networks.
    
    This module implements scaled dot-product attention across all nodes in a graph,
    allowing global information exchange beyond local neighborhoods. Each node can
    attend to all other nodes in the graph to gather global context.
    
    Args:
        hidden_dim (int): Dimension of node embeddings.
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = Linear(hidden_dim, hidden_dim)
        self.key = Linear(hidden_dim, hidden_dim)
        self.value = Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        
    def forward(self, x, batch=None):
        """
        Forward pass for global attention.
        
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, hidden_dim].
            batch (torch.Tensor, optional): Batch indices for batched graphs.
                If None, all nodes are treated as belonging to one graph.
        
        Returns:
            torch.Tensor: Attention-weighted node features of shape [num_nodes, hidden_dim].
        """
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
