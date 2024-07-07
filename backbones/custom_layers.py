import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Maxmin Activation

class MaxMin(nn.Module):
    def __init__(self):
        super(MaxMin, self).__init__()

    def forward(self, z, axis=1):
        a, b = z.split(z.shape[axis] // 2, axis)
        c, d = torch.max(a, b), torch.min(a, b)
        return torch.cat([c, d], dim=axis)

# SLL [ICLR'23]
# Forked from https://github.com/araujoalexandre/lipschitz-sll-networks

class SLLConv2d(nn.Module):
    def __init__(self, cin, inner_dim, kernel_size=3, stride=1, bias = False):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else cin
        self.activation = nn.ReLU()

        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(inner_dim, cin, kernel_size, kernel_size))
        self.bias = None
        if bias == True:
            self.bias = nn.Parameter(torch.empty(1, inner_dim, 1, 1))
            nn.init.orthogonal_(self.weight)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound) # bias init            
        self.q = nn.Parameter(torch.randn(inner_dim))


    def compute_t(self):
        ktk = F.conv2d(self.weight, self.weight, padding=self.weight.shape[-1] - 1)
        ktk = torch.abs(ktk)
        q = torch.exp(self.q).reshape(1, -1, 1, 1)
        q_inv = torch.exp(-self.q).reshape(-1, 1, 1, 1)
        t = (q_inv * ktk * q).sum((1, 2, 3))
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        t = t.reshape(1, -1, 1, 1)
        res = F.conv2d(x, self.weight, padding=1)
        if self.bias != None:
            res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.conv_transpose2d(res, self.weight, padding=1)
        out = x - res
        return out

class SLLLinear(nn.Module):
    def __init__(self, ic, oc, bias = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(oc, ic))
        self.bias = None
        if bias == True:
            self.bias = nn.Parameter(torch.empty(1, oc))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound) # bias init          
        self.q = nn.Parameter(torch.randn(oc))
        nn.init.orthogonal_(self.weight)

    def compute_t(self):
        # oc * oc
        wwt = torch.matmul(self.weight, self.weight.T)
        wwt = torch.abs(wwt)
        q = torch.exp(self.q).reshape(-1, 1)
        q_inv = torch.exp(-self.q).reshape(-1, 1)
        t = (q_inv * wwt * q).sum(1)
        t = safe_inv(t).sqrt()
        return t
        
    def forward(self, x):
        t = self.compute_t()
        x = F.linear(x, self.weight, self.bias)
        return t * x
    
def safe_inv(x):
    mask = x == 0
    x_inv = x**(-1)
    x_inv[mask] = 0
    return x_inv
    
# PaddingChannels
    
class PaddingChannels(nn.Module):

    def __init__(self, ncout, ncin=3, mode="zero"):
        super().__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.mode = mode

    def forward(self, x):
        if self.mode == "clone":
            return x.repeat(1, int(self.ncout / self.ncin), 1, 1) / np.sqrt(int(self.ncout / self.ncin))
        elif self.mode == "zero":
            bs, _, size1, size2 = x.shape
            out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
            out[:, :self.ncin] = x
            return out