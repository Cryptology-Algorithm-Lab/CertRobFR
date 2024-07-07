# Backbone we used in whole experiments.

import torch
from torch import nn
import torch.nn.functional as F
from backbones.custom_layers import SLLConv2d, PaddingChannels, SLLLinear, MaxMin

# ConvBlock

class SLLBlock(nn.Module):
    def __init__(self, nc, n_blk):
        super().__init__()
        layer = []
        
        for _ in range(n_blk):
            layer.append(SLLConv2d(nc, nc, kernel_size= 3, bias = True))

        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.layer(x)

class FirstChannels(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.nc = nc
    
    def forward(self, x):
        return x[:, :self.nc, :, :]

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
    
# Main SLLNet; we use config = [3,4,6,3]
class SLLNet(nn.Module):
    def __init__(self, config, emb_size = 512, fp16 = False):
        super().__init__()
        width = 1
        self.stem = nn.Sequential(
            PaddingChannels(32),
            SLLConv2d(32, 32, kernel_size = 3, bias = True),
            Downsize(32,64)
        )
        
        self.layer1 = SLLBlock(64, config[0])
        self.pool1 = Downsize(64, 128 * width)        
        self.layer2 = SLLBlock(128 * width, config[1])
        self.pool2 = Downsize(128 * width, 256 * width) 
        self.layer3 = SLLBlock(256 * width, config[2])
        self.pool3 = Downsize(256 * width, 512) 
        self.layer4 = SLLBlock(512, config[3])
        self.linear = SLLLinear(512 * 49, emb_size)
        self.fp16 = fp16
 
    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.pool1(x)
            x = self.layer2(x)
            x = self.pool2(x)
            x = self.layer3(x)
            x = self.pool3(x)
            x = self.layer4(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
        return x.float() if self.fp16 else x