# Codes for reproducing toy examples in Section 3.2
import time
import torch
from torch import nn
import torch.nn.functional as F

from headers import ECosFace
from torch.utils.data import DataLoader

def train(cfg, Model):
    backbone = Model()
    header = ECosFace(10, 512)
    
    backbone.eval().to(cfg.device)
    header.eval().to(cfg.device)
    
    train_loader = DataLoader(cfg.train_dataset, cfg.batch_size, shuffle = True, num_workers = 8)
    loss_fn = nn.CrossEntropyLoss()
    
    optim = torch.optim.AdamW(
        [{"params": backbone.parameters()}, {"params": header.parameters()}],
        lr = cfg.lr,
        weight_decay = cfg.wd,
    )
    
    mem_acc = []
    mem_fn = []
    
    for i in range(cfg.epoch):
        t1 = time.time()
        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            
            embs = backbone(x)
            logits = header(embs, y)
            train_loss = loss_fn(logits, y)
            reg_loss = torch.zeros(1, device = cfg.device) if cfg.xi ==0 else F.relu(- cfg.xi * torch.log(embs.norm(p=2, dim=1) / cfg.xi)).mean()
            loss = train_loss + cfg.lambd * reg_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            
        acc, avg_fn = benchmark(cfg, backbone, header)
        elapsed = time.time() - t1
        
        print(f"[{i+1}/{cfg.epoch}] acc: {acc}, avg_fn: {avg_fn}, time: {elapsed}")
        mem_acc.append(acc)
        mem_fn.append(avg_fn)
        backbone.train()
        header.train()
        
        
    return backbone, header, mem_acc, mem_fn

@torch.no_grad()
def benchmark(cfg, backbone, header):
    test_loader = DataLoader(cfg.test_dataset, cfg.batch_size, shuffle = False, num_workers = 8)
    backbone.eval()
    header.eval()
    
    header_weight = F.normalize(header.weight.data)
    
    n_samples = 0
    matches = 0
    fn_sum = 0
    for x, y in test_loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        embs = backbone(x)
        logits= torch.mm(embs, header_weight.T)
        pred = logits.argmax(dim=1)
        
        n_samples += x.size(0)
        matches += (y == pred).sum().item()
        fn_sum += embs.norm(p=2,dim=1).sum().item()
        
        
    avg_fn = fn_sum / n_samples
    acc = matches / n_samples
        
    return acc, avg_fn