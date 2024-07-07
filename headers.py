# Headers

import torch
from torch import nn
import torch.nn.functional as F
import math    

# ArcFace header
# Forked from: https://github.com/deepinsight/insightface

class ArcFace(nn.Module):
    def __init__(self, num_classes, emb_size, s = 64.0, m = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, emb_size) * 0.01)
        self.s = s
        self.m = m    
    
    def forward(self, x, labels):
        weight = F.normalize(self.weight)
        x = F.normalize(x)
        logits = F.linear(x, weight, None)
        
        with torch.no_grad():
            index = torch.where(labels != -1)[0]
            target_logit = logits[index, labels[index].view(-1)]
            target_theta = torch.acos(target_logit)
            final_target_logit = target_theta + self.m
            logits[index, labels[index].view(-1)] = torch.cos(final_target_logit)
        return self.s * logits   
    
# AdaCos header
    
class AdaCos(nn.Module):
    def __init__(self, num_classes, emb_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, emb_size) * 0.01)
        self.s =  math.log2((num_classes - 1)) * math.sqrt(2)
        
    def forward(self, x, labels):
        weight = F.normalize(self.weight)
        x = F.normalize(x)
        logits = F.linear(x, weight)
        with torch.no_grad():
            index = torch.where(labels != -1)[0]
            B_avg = torch.exp(self.s * logits)
            B_avg[index, labels[index].view(-1)] = 0
            B_avg = B_avg.sum() / logits.size(0)
            target_logit = logits[index, labels[index].view(-1)]
            target_theta = torch.acos(target_logit.clamp(-1, 1))
            theta_med = torch.median(target_theta)
            self.s = torch.log(B_avg + 1e-6) / (torch.cos(torch.min(torch.pi/4 * torch.ones_like(theta_med), theta_med)) + 1e-6)
        
        return self.s * logits  

# ElasticFace-Cos
# Forked from: https://github.com/fdbtrs/ElasticFace

class ECosFace(nn.Module):
    def __init__(self, num_classes, emb_size, s=64.0, m=0.35, std=0.025, plus=True):
        super().__init__()
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, emb_size) * 0.01)
        self.std=std
        self.plus=plus

    def forward(self, x, labels):
        weight = F.normalize(self.weight)
        x = F.normalize(x)
        logits = F.linear(x, weight)
        index = torch.where(labels != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device)
        margin = torch.normal(mean=self.m, std=self.std, size=labels[index, None].size(), device=logits.device)  # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = logits[index, labels.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, labels[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, labels[index, None], margin)
        logits[index] -= m_hot
        ret = logits * self.s
        return ret        
    
# Usual Header with LLN Technique

class NaiveHeader(nn.Module):
    def __init__(self, num_classes, emb_size, s = 4, m = 8 **.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, emb_size) * 0.01)
        self.m = m
        self.s = s
        
    def forward(self, x, labels):
        weight = F.normalize(self.weight)
        logits = F.linear(x, weight)
        index = torch.where(labels != -1)[0]
        with torch.no_grad():
            target_logit = logits[index, labels[index].view(-1)]
            final_target_logit = target_logit - self.m
            logits[index, labels[index].view(-1)] = final_target_logit
        return logits * self.s