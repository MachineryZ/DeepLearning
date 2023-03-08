# DDPM

Denoising Diffusion Probabilistic Models
https://arxiv.org/pdf/2006.11239.pdf

<div align=center><img src="../Files/ddpm.jpg" width=90%></div>

ddpm 代码：
~~~python
# train.py
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from torchvision import transforms, datasets

from model import Unet



~~~

model.py:
~~~python

import math
from inspect import isfunction
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

def exisit(x):
    return x is not None

def default(val, d):
    if exisit(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, 1, 1)
    )

def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2)) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1, 1, 0),
    )


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        return embeddings

~~~