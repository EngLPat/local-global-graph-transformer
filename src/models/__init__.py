"""
FCLGA GraphTransformer Models.

This package contains the model architecture for the Frequency-Controlled
Layered Global Attention GraphTransformer.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

from .fclga_graph_transformer import FCLGA_GraphTransformer
from .processor_layer import ProcessorLayer
from .attention import GlobalAttention

__all__ = ['FCLGA_GraphTransformer', 'ProcessorLayer', 'GlobalAttention']
