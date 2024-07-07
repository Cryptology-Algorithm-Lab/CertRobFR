# Attacks including FGSM, PGD, and C&W
# All attacks are defined in the l2 norm.

import torch
from torch import nn
import torch.nn.functional as F

# Abstract Attack Engine
class AttackEngine:
    def __init__(self, backbone, device, **kwargs):
        self.backbone = backbone
        self.device = device
        self.backbone.eval().to(device)
    
    ### Attack Algorithm ###
    def attack_untargeted(self):
        NotImplemented        
        
    def attack_targeted(self):
        NotImplemented
        

        
class FGSM(AttackEngine):
    def __init__(self, backbone, device, params):
        super().__init__(backbone, device)
        self.params = params
        self.type = "FGSM"
        self.eps = params.eps
        self.backbone = backbone
        self.device = device    
    
    ### ATTACK ALGORITHMS ###    
    def attack_untargeted(self, img):
        img_ = img.clone()
        img_.requires_grad_(True)
        feat = self.backbone(img_.to(self.device))
        cosine = F.cosine_similarity(feat, feat).mean()
        grad = torch.autograd.grad(cosine, img_)[0]
        delta = self.eps * (grad / (grad.norm(p=2, dim=[1,2,3], keepdim = True)))
        adv_img = (img_ - delta).clamp(-.5, .5)
        return adv_img
    
    
    def attack_targeted(self, img, tg_feat, mode = "D"):
        img.requires_grad_(True)
        feat = self.backbone(img.to(self.device))
        cosine = F.cosine_similarity(feat, tg_feat.to(self.device)).mean()
        grad = torch.autograd.grad(cosine, img)[0]
        delta = self.eps * (grad / (grad.norm(p=2, dim=[1,2,3], keepdim = True) + 1e-6))
        if mode == "D":
            adv_img = img - delta
        else:
            adv_img = img + delta
        return adv_img.clamp(-0.5, 0.5).detach()    
    
    def __repr__(self):
        ret = "FGSM Adversary with the following parameters\n"
        ret+= "============================================\n"
        for name in self.params:
            ret+= f"{name}:\t\t{self.params[name]}\n"
        return ret
    
    
class PGD(AttackEngine):
    def __init__(self, backbone, device, params):
        super().__init__(backbone, device)
        self.params = params
        self.type = "PGD"
        self.eps = params.eps
        self.alpha = params.alpha * params.eps / params.n_iter
        self.n_iter = params.n_iter
        self.backbone = backbone
        self.device = device    
    
    ### ATTACK ALGORITHMS ###
    def attack_untargeted(self, img):
        orig_img = img.clone().to(self.device)
        adv_img = img.clone().to(self.device)
        orig_feat = self.backbone(img.to(self.device))
        
        for _ in range(self.n_iter):
            adv_img.requires_grad_(True)
            feat = self.backbone(adv_img)
            cosine = F.cosine_similarity(feat, orig_feat).mean()
            grad = torch.autograd.grad(cosine, adv_img)[0]
            
            with torch.no_grad():
                adv_img = adv_img - self.alpha * (grad / (grad.norm(p=2, dim=[1,2,3], keepdim = True) + 1e-6))
                delta = adv_img - orig_img
                delta = delta.renorm(p=2,dim=0, maxnorm = self.eps)
                adv_img = delta + orig_img
                adv_img.clamp(-0.5, 0.5)
        return adv_img.clamp(-0.5, 0.5).detach()

    def attack_targeted(self, img, tg_feat, mode = "D"):
        orig_img = img.clone().to(self.device)
        adv_img = img.clone().to(self.device)
        for _ in range(self.n_iter):
            adv_img.requires_grad_(True)
            feat = self.backbone(adv_img)
            cosine = F.cosine_similarity(feat, tg_feat).mean()
            grad = torch.autograd.grad(cosine, adv_img)[0]
            
            with torch.no_grad():
                if mode == "D":
                    adv_img = adv_img - self.alpha * (grad / (grad.norm(p=2, dim=[1,2,3], keepdim = True) + 1e-6))
                else:
                    adv_img = adv_img + self.alpha * (grad / (grad.norm(p=2, dim=[1,2,3], keepdim = True) + 1e-6))
                delta = adv_img - orig_img
                delta = delta.renorm(p=2,dim=0, maxnorm = self.eps)
                adv_img = delta + orig_img
                adv_img.clamp(-0.5, 0.5)
                
        return adv_img.clamp(-0.5, 0.5).detach()    
    
    
    def __repr__(self):
        ret = "PGD Adversary with the following parameters\n"
        ret+= "===========================================\n"
        for name in self.params:
            ret+= f"{name}:\t\t{self.params[name]}\n"
        return ret    
    
    
class CW(AttackEngine):
    def __init__(self, backbone, device, params):
        super().__init__(backbone, device)
        self.params = params
        self.type = "CW"
        self.alpha = params.alpha
        self.n_iter = params.n_iter
        self.kappa1 = params.kappa1
        self.kappa2 = params.kappa2
        self.c = params.c
        
    ### ATTACK ALGORITHMS ###            
    def attack_untargeted(self, img):        
        orig_img = img.clone().to(self.device)
        orig_feat = self.backbone(img.to(self.device))
        adv_w = torch.atanh(2 * img.clone().clamp(-.5 + 1e-6, .5 - 1e-6)).to(self.device)
        adv_w.requires_grad_(True)
        optim = torch.optim.Adam([adv_w], lr = self.alpha)

        for _ in range(self.n_iter):
            adv_img = torch.tanh(adv_w) * (1/2)
            feat = self.backbone(adv_img)
            loss = F.relu((adv_img - orig_img).norm(p=2, dim=[1,2,3]) - self.kappa2).mean() + self.c * F.relu(F.cosine_similarity(feat, orig_feat) - self.kappa1).mean()
            optim.zero_grad()
            loss.backward(retain_graph = True)
            optim.step()
                        
        adv_img = torch.tanh(adv_w) * (1/2)
        return adv_img.detach()

    def attack_targeted(self, img, tg_feat, mode = "D"):        
        orig_img = img.clone().to(self.device)
        adv_w = torch.atanh(2 * img.clone().clamp(-.5 + 1e-6, .5 - 1e-6)).to(self.device)
        adv_w.requires_grad_(True)
        optim = torch.optim.Adam([adv_w], lr = self.alpha)
        
        for _ in range(self.n_iter):
            adv_img = torch.tanh(adv_w) * (1/2)
            feat = self.backbone(adv_img)
            cosine = F.cosine_similarity(feat, tg_feat)
            if mode == "D":
                loss = (adv_img - orig_img).norm(p=2, dim=[1,2,3]).mean() + self.c * F.relu(F.cosine_similarity(feat, tg_feat) - self.kappa1).mean()
            else:
                loss = (adv_img - orig_img).norm(p=2, dim=[1,2,3]).mean() + self.c * F.relu(1 - F.cosine_similarity(feat, tg_feat) - self.kappa1).mean()
            optim.zero_grad()
            loss.backward(retain_graph = True)
            optim.step()
        
        adv_img = torch.tanh(adv_w) * (1/2)
        return adv_img.detach().clamp(-.5, .5)
                
    
    def __repr__(self):
        ret = "C&W Adversary with the following parameters\n"
        ret+= "===========================================\n"
        for name in self.params:
            ret+= f"{name}:\t\t{self.params[name]}\n"
        return ret        