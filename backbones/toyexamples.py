# Networks for Toy Examples
from backbones.custom_layers import SLLConv2d, PaddingChannels, SLLLinear, MaxMin
import torch
from torch import nn
import torch.nn.functional as F

# Downsize
class Downsize(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.act = MaxMin()
        self.first = FirstChannels(oc // 4)
        self.unshuffle = nn.PixelUnshuffle(2)
        
    def forward(self, x):
        x = self.act(x)
        x = self.first(x)
        x = self.unshuffle(x)
        return x
    
# Take first channels    
class FirstChannels(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.nc = nc
    
    def forward(self, x):
        return x[:, :self.nc, :, :]    
    
    
# Block for SLLNet
class SLLBlock(nn.Module):
    def __init__(self, nc, n_blk):
        super().__init__()
        layer = []
        for _ in range(n_blk):
            layer.append(SLLConv2d(nc, nc, 3, bias = True))
    
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.layer(x)
    
# Block for usual ConvNet
class ConvBlock(nn.Module):
    def __init__(self, nc, n_blk):
        super().__init__()
        layer = []
        for _ in range(n_blk):
            layer.append(nn.Conv2d(nc, nc, 3, 1, 1))
            layer.append(nn.BatchNorm2d(nc))
            layer.append(nn.ReLU())
    
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.layer(x)
    
    
class SLLNet_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            PaddingChannels(16, 1),
            SLLConv2d(16, 16, 1, bias = True),
            Downsize(16, 32)
        )
        
        self.body = nn.Sequential(
            SLLBlock(32, 3),
            Downsize(32, 64),
            SLLBlock(64, 3),
        )
        
        self.neck = nn.Sequential(
            nn.Flatten(),
            SLLLinear(7 * 7 * 64, 512, bias = False),
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.neck(x)
        return x

class SLLNet_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            PaddingChannels(32),
            SLLConv2d(32, 32, 3, bias = True),
            Downsize(32, 64)
        )
        
        self.body = nn.Sequential(
            SLLBlock(64, 1),
            Downsize(64, 128),
            SLLBlock(128, 3),
            Downsize(128, 256),
            SLLBlock(256, 1),            
        )
        
        self.neck = nn.Sequential(
            nn.Flatten(),
            SLLLinear(4 * 4 * 256, 512, bias = False),
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.neck(x)
        return x



class Net_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU()
        )
        
        self.body = nn.Sequential(
            ConvBlock(32, 3),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            ConvBlock(64, 3),
        )
        
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512, bias = False)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.neck(x)
        return x


class Net_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),            
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.body = nn.Sequential(
            ConvBlock(64, 1),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ConvBlock(128, 3),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ConvBlock(256, 1),            
        )
        
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 512, bias = False)
        )
    
        
    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.neck(x)
        return x

