"""
Optimizer Utilities for FCLGA GraphTransformer.

This module provides functions for creating optimizers and learning rate schedulers
with various configuration options.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import math
import torch.optim as optim


def build_optimizer(args, params):
    """
    Build optimizer and learning rate scheduler based on configuration.
    
    Creates an optimizer (Adam, SGD, RMSprop, or Adagrad) and optionally
    a learning rate scheduler (step decay, cosine annealing, or cosine with warmup).
    
    Args:
        args: Configuration object with optimizer parameters including:
            - opt: Optimizer type ('adam', 'sgd', 'rmsprop', 'adagrad')
            - lr: Learning rate
            - weight_decay: Weight decay (L2 regularization)
            - opt_scheduler: Scheduler type ('none', 'step', 'cos', 'cosine')
            - opt_decay_step: Step size for step scheduler
            - opt_decay_rate: Decay rate for step scheduler
            - opt_restart: T_max for cosine annealing
            - epochs: Total epochs for cosine warmup scheduler
        params: Model parameters to optimize.
    
    Returns:
        tuple: (scheduler, optimizer) where scheduler may be None if not used.
    """
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
