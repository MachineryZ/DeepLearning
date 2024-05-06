import torch
import torch.nn as nn
import numpy as np
from .spline import *


class KANLayer(nn.Module):
    """
    KANLayer class

    Attributes:
    in_dim (int): input dimension
    out_dim (int): output dimension
    size (int): the number of splines = input dimension * output dimension
    k (int): the piecewise polynomial order of splines
    grid (2D torch.float): grid points
    noises (2D torch.float): coefficients of B-spline bases
    coef (2D torch.float): injected noises to splines at initialization (to break degeneracy)
    scale_base (1D torch.float): magnitude of the residual function b(x)
    scale_sp (1D torch.float): magnitude of the spline function spline(x)
    base_fun (fun): residual function b(x)
    mask (1D torch.float): mask of spline functions. setting some element of the mask to zero 
        means setting the corresponding activation to zero function.
    grid_eps (float in [0,1]): a hyperparameter used in update_grid_from_samples.
        When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using
        percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. 
    weight_sharing (1D tensor int): allow spline activations to share parameters
    lock_counter (int): counter how many acivation functions are locked (weight sharing)
    lock_id (1D torch.int): the id of activation functions that are locked
    device (str): device

    Methods:
    __init__: initialize a KANLayer
    forward: forward
    update_grid_from_samples: update grids based on samples' incoming activations
    initialize_grid_from_parent: initialize grids from another model
    get_subset: get subset of the KANLayer (used for pruning)
    lock: lock several activation functions to share parameters
    unlock: unlock already locked activation functions
    """
    def __init__(
        self,
        in_dim=3,
        out_dim=2,
        num=5,
        k=3,
        noise_scale=0.1,
        scale_base=1.0,
        scale_sp=1.0,
        base_fun=torch.nn.SiLU(),
        grid_eps=0.02,
        grid_range=[-1, 1],
        sp_trainable=True,
        sb_trainable=True,
        device='cpu',
    ) -> None:
        """ Initialize a KANLayer
        Args:
            in_dim (int): input dimension. Default: 2
            out_dim (int): output dimension. Default: 3
            num (int): the number of grid intervals = G. Default: 5.
            k (int): the order of piecewise polynomial. Default: 3
            noise_scale (float): the scale of noise injected at initialization. Default: 0.1
            scale_base (float): the scale of the residual function b(x). Default: 1.0
            scale_sp (float): the scale of the base function spline(x). Default: 1.0
            base_fun (function): residual function b(x). Default: torch.nn.SiLU()
            grid_eps (float): When grid_eps = 0, the grid is uniform; when grid_eps = 1,
                the grid is partitioned using percentiles of samples. 0 < grid_eps < 1
                interpolates between the two extremes. Default 0.02
            grid_range (List/np.array of shape (2,)): setting range of grids. Default: [-1, 1]
            sp_trainable (bool): if true, scale_sp is trainable. Default: True
            sb_trainable (bool): if true, scale_base is trainable. Default: True
            device (str): device
        Returns:
            self
        """
        super(KANLayer, self).__init__()
        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # shape: (size, num)
        self.grid = torch.einsum('i,j->ij', 
                                 torch.ones(size, device=device), 
                                 torch.linspace(grid_range[0], grid_range[1], steps=num+1, device=device))
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        noises = (torch.rand(size, self.grid.shape[1]) - 1/2) * noise_scale / num
        noises = noises.to(device)
        # shape: (size, coef)
        self.coef = torch.nn.Parameter(curve2coef(self.grid, noises, self.grid, k, device))
        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(torch.ones(size, device=device) * scale_base).requires_grad_(sb_trainable)
        else:
            self.scale_base = torch.nn.Parameter(torch.FloatTensor(scale_base).cuda()).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(size, device=device)).requires_grad_(False)
        self.base_fun = base_fun

        self.mask = torch.nn.Parameter(torch.ones(size, device=device)).requires_grad_(False)
        self.grid_eps = grid_eps
        self.weight_sharing = torch.arange(size)
        self.lock_counter = 0
        self.lock_id = torch.zeros(size)
        self.device = device

    def forward(self, x):
        """ KANLayer forward given input x
        Args:
            x (2D torch.float): inputs, shape (number of samples, input dimension)
        Returns:
            y (2D torch.float): outputs, shape (number of samples, output dimension)
            preacts (3D torch.float): fan out x into activations, shape (number of samples, output dimension, input dimension)
            postacts (3D torch.float): the outputs of activation functions with preacts as inputs
            postspline (3D torch.float): the outputs of spline functions with preacts as inputs
        """
        