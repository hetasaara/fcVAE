# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:14:07 2020

@author: hiltunh3
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


class MultiInputLayer(nn.Module):
    """Multiple-input layer.
    
    :param d_shared: number of shared input dimensions
    :param d_other: number of domain-specific input dimensions (same for all domains)
    :param n_other: number of domains
    :param d_out: number of output dimensions
    """
    def __init__(self, d_shared: int, d_other: int, n_other: int, d_out: int):
        super(MultiInputLayer, self).__init__()
        self.bias_shared = torch.nn.Parameter(torch.Tensor(d_out))
        self.bias_other = torch.nn.Parameter(torch.Tensor(d_out, n_other))
        self.weight_shared = torch.nn.Parameter(torch.Tensor(d_out, d_shared))
        self.weight_other = torch.nn.Parameter(torch.Tensor(d_out, d_other, n_other))
        self.init_params()
        
    def init_params(self):
        """Initializes the parameters."""
        init.kaiming_uniform_(self.weight_shared, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_other, a=math.sqrt(5))
        a1, _ = init._calculate_fan_in_and_fan_out(self.weight_shared)
        a2, _ = init._calculate_fan_in_and_fan_out(self.weight_other)
        b1 = 1 / math.sqrt(a1)
        b2 = 1 / math.sqrt(a2)
        init.uniform_(self.bias_shared, -b1, b1)
        init.uniform_(self.bias_other, -b2, b2)
        
    def forward(self, x_shared: torch.Tensor, x_other: torch.Tensor, idx: int):
        """Forward pass through the layer.
        
        :param x_shared: Shared data. Tensor with shape [n_batch, d_shared].
        :param x_other: Domain-specific data. Tensor with shape [n_batch, d_other].
        :param idx: Domain index. Must be an integer in {0, ..., self.n_other-1}.
        
        :return: Tensor with shape [n_batch, d_out].
        """
        
        h_1 = F.linear(x_shared, self.weight_shared, self.bias_shared)
        h_2 = F.linear(x_other, self.weight_other[:,:,idx], self.bias_other[:,idx])
        x_out = h_1 + h_2
        return x_out