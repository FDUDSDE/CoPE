import math
import numpy as np
# import scipy as sc
from scipy import special

import torch
from torch import nn
from torch.nn import functional as F


class InvNet(nn.Module):
    
    def __init__(self, order):
        super().__init__()
        self.order = order
    
    def forward(self, A, x, alpha=1.):
        zs = [x]
        z = x
        for _ in range(self.order):
            z = alpha * (A @ z)
            zs.append(z)
        return torch.stack(zs, 0).sum(0)

    
class ExpNet(nn.Module):
    
    def __init__(self, order):
        super().__init__()
        self.order = order
        self.coefs = self.compute_bessel_coefs(order)
    
    def compute_bessel_coefs(self, order):
        coefs = special.jv(np.arange(order+1), 0-1j) * (0+1j) ** np.arange(order+1)
        coefs = 2 * coefs.real
        coefs[0] /= 2
        return torch.from_numpy(coefs).float()
    
    def forward(self, A, x, alpha=1.):
        pp_state = x
        p_state = alpha * (A @ x)
        zs = [pp_state, p_state]
        for _ in range(self.order-1):
            new_state = 2 * alpha * (A @ p_state) - pp_state
            zs.append(new_state)
            pp_state, p_state = p_state, new_state
        return (torch.stack(zs, 0) * self.coefs.to(x.device).reshape(-1, 1, 1)).sum(0)

    
class ACGNN(nn.Module):
    
    def __init__(self, inv_order, exp_order, n_nodes, learnable_alpha=False):
        super().__init__()
        self.inv_net = InvNet(inv_order)
        self.exp_net = ExpNet(exp_order)
        self.n_nodes = n_nodes
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.ones(n_nodes) * 3)
        else:
            self.register_buffer('alpha', torch.ones(n_nodes))
    
    def forward(self, A, init_state, last_state, t):
        d = last_state.size(1)
        if self.learnable_alpha:
            alpha = torch.sigmoid(self.alpha)
        else:
            alpha = self.alpha
        alpha = alpha.unsqueeze(1)
        # e^{(A_I)t} x
        scale = math.ceil(t)
        z = torch.cat([init_state, last_state], 1) * math.exp(-t)
        for _ in range(scale):
            z = self.exp_net(A / scale, z, alpha)
        init_exp, last_exp = torch.split(z, d, 1)
        # (A-I)^{-1} (x - e^{(A_I)t} x)
        init_inv = self.inv_net(A, init_state - init_exp, alpha)
        return init_inv + last_exp
