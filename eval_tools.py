# Evaluation Tools

import torch
from torch import nn
import torch.nn.functional as F

### Basic Tools

@torch.no_grad()
def feat_ext(backbone, dataset,batch_size, device):
    backbone = backbone.eval().to(device)
    data = dataset[0]
    outs = []
    feat_norm_acc = 0
    for dset in data:
        start = 0
        tmp = []
        while start < dset.size(0):
            end = min(start + batch_size, dset.size(0))
            bb = (dset[start:end].to(device) - 127.5) / 255
            feat = backbone(bb)
            feat_norm = feat.norm(2, dim = 1).sum()
            feat_norm_acc += feat_norm.item()
            feat = F.normalize(feat)
            tmp.append(feat)
            start = end
        outs.append(torch.cat(tmp, dim = 0))
        
    mean_feat_norm = feat_norm_acc / (outs[0].size(0) + outs[1].size(0))
#     final_feat = F.normalize(outs[0] + outs[1])
    final_feat = torch.cat(outs, dim = 0)
        
    return final_feat, mean_feat_norm

# Unflipped Version            
@torch.no_grad()
def benchmark_standard(feats, target_far = None):
    left, right = feats[::2], feats[1::2]
    num_pairs = left.size(0)
    stepsize = left.size(0) // 40
    issame = []
    
    for i in range(num_pairs):
        if (i // stepsize) % 2 == 0:
            issame += [True]
        else:
            issame += [False]
    
    
    score = F.cosine_similarity(left, right)
    issame = torch.BoolTensor(issame).to(feats.device)
    
    taus = torch.linspace(-1, 1, 1000)

    if target_far == None:
        # Choose The Threshold for the best accuracy    
        best_acc = 0
        best_thres = 0
        best_tar = 0
        best_far = 0
        
        for tau in taus:
            pred = (score > tau)
            TA = (pred & issame).sum()
            TR = (~pred & ~issame).sum()
            FA = (pred & ~issame).sum()
            FR = (~pred & issame).sum()
            ACC = (TA + TR) / num_pairs
            if ACC > best_acc:
                best_acc = ACC.item()
                best_tar = TA.item()  / (num_pairs / 2)
                best_far = FA.item()  / (num_pairs / 2)
                best_thres = tau
            
        return best_tar, best_far, best_acc, best_thres


    else:
        # Choose The Threshold for the target FAR
        for tau in reversed(taus):
            pred = (score > tau)
            TA = (pred & issame).sum()
            TR = (~pred & ~issame).sum()
            FA = (pred & ~issame).sum() 
            FR = (~pred & issame).sum()
            ACC = (TA + TR) / num_pairs    
            
            if FA / (num_pairs / 2) > target_far:
                return TA  / (num_pairs / 2), FA  / (num_pairs / 2), ACC, tau
            
@torch.no_grad()
def benchmark_certified(dataset, backbone, batch_size, device, tau, eps):
    backbone = backbone.to(device)
   
    # Untargeted Attack (Type 1)
    data, _ = dataset
    unique_dataset = torch.cat(data,dim=0).unique(dim=0)
    n_dset = unique_dataset.size(0)
    print(f"Total # of Unique Images: {n_dset}")
    
    unt_score = 0
    start = 0
    mem1 = []
    while start < n_dset:
        end = min(start + batch_size, n_dset)
        blk = (unique_dataset[start:end] - 127.5) / 255
        emb = backbone(blk.to(device))
        cert = (emb.norm(2, dim = 1) * ((1 - tau**2)**.5)) > eps
        unt_score += cert.sum().item()
        start = end
        
        mem1.append((emb.norm(2, dim = 1) * ((1 - tau**2)**.5)))
        
    mem1 = torch.cat(mem1)
    e_ut = torch.median(mem1).item()
        
    unt_acc = unt_score / n_dset    
    
    # Targeted Attack
    t_score = 0
    
    mem2 = []
    
    for dset in data:
        # Only Use False Pairs!
        left, right = dset[::2], dset[1::2]
        left_neg = left.reshape(20, -1, 3, 112, 112)[1::2,:,:,:,:].reshape(-1,3,112,112)
        right_neg = right.reshape(20, -1, 3, 112, 112)[1::2,:,:,:,:].reshape(-1,3,112,112)
        
        n_nset = left_neg.size(0)
        start = 0
        
        while start < n_nset:
            end = min(start + batch_size, n_nset)
            l_blk = (left_neg[start:end] - 127.5) / 255
            r_blk = (right_neg[start:end] - 127.5) / 255
            l_emb = backbone(l_blk.to(device))
            r_emb = backbone(r_blk.to(device))
            min_norm = torch.minimum(l_emb.norm(2, dim = 1), r_emb.norm(2, dim=1))
            
            cos = F.cosine_similarity(l_emb, r_emb)
            sin = (1-cos.pow(2)).clamp(0, 1).sqrt()
            cert1 = min_norm * (sin * tau - cos * ((1 - tau ** 2) ** .5)) > eps
            cert2 = cos < tau
            cert = cert1 & cert2
            t_score += cert.sum().item()
            start = end
            
            eps_d = min_norm * (sin * tau - cos * ((1 - tau ** 2) ** .5))
            eps_d = torch.where(eps_d>0, eps_d, 0)
            
            mem2.append(eps_d)
    
    mem2 = torch.cat(mem2)
    e_t = torch.median(mem2).item()
    
    t_acc = t_score / (n_nset * 2)
    
    # Untargeted Attack (Type 2)
    unt2_score = 0
    
    mem3 = []
    
    for dset in data:
        # Only Use True Pairs!
        left, right = dset[::2], dset[1::2]
        left_pos = left.reshape(20, -1, 3, 112, 112)[::2,:,:,:,:].reshape(-1,3,112,112)
        right_pos = right.reshape(20, -1, 3, 112, 112)[::2,:,:,:,:].reshape(-1,3,112,112)
        
        n_pset = left_neg.size(0)
        start = 0
        
        while start < n_pset:
            end = min(start + batch_size, n_nset)
            l_blk = (left_pos[start:end] - 127.5) / 255
            r_blk = (right_pos[start:end] - 127.5) / 255
            l_emb = backbone(l_blk.to(device))
            r_emb = backbone(r_blk.to(device))
            min_norm = torch.minimum(l_emb.norm(2, dim = 1), r_emb.norm(2, dim=1))
            
            cos = F.cosine_similarity(l_emb, r_emb)
            sin = (1-cos.pow(2)).clamp(0, 1).sqrt()
            cert1 = min_norm * (-sin * tau + cos * ((1 - tau ** 2) ** .5)) > eps
            cert2 = cos > tau
            cert = cert1 & cert2
            unt2_score += cert.sum().item()
            start = end        
            
            eps_i = min_norm * (-sin * tau + cos * ((1 - tau ** 2) ** .5))
            eps_i = torch.where(eps_i>0, eps_i, 0)            
            
            mem3.append(eps_i)
    
    mem3 = torch.cat(mem3)
    e_ut2 = torch.median(mem3).item()
    
    unt2_acc = unt2_score / (n_pset * 2)
        
    
    return unt_acc, unt2_acc, t_acc, e_ut, e_ut2, e_t

# PGD Benchmark Function
def test_pgd(backbone, datasets, mode, device, attack_configs, batch_size = 128, suffix = None, target_far = 1e-3, **kwargs):
    print(suffix)
    exp_name = suffix.strip()[0]
    # Reporting Empirical Robustness
    if mode == "E":
        print("Evaluating Empirical Robustness.")
        print("Initializing Attack Engines...\n\n")
        
        attack_engines = []
        
        for name, alg, param in attack_configs: 
            attack_engines.append((name, alg(backbone, device, param)))
        
        # Display Attack Algorithms
        print("<<< ATTACK ALGORITHMS >>>")
        for name, engine in attack_engines:
            print("Mode: ", name)
            print(engine)
        
    out = []
    for dset_name, dataset in datasets:
        print(f"Evalaution on Dataset: {dset_name}")
        # Run Standard Benchmark for each dataset
        feat, mfeat_norm = feat_ext(backbone, dataset, batch_size, device)
        tar, far, acc, tau = benchmark_standard(feat, target_far = target_far)
        
        print(f"TAR: {tar:.4f}\tFAR: {far:.4f}\t ACC: {acc:.4f}\t TAU: {tau:.4f}\tAvg. FN: {mfeat_norm:.4f}")
        
        for name, engine in attack_engines:
            print(f"Evaluating Empirical Robustness for {engine.type}")
            rets = pgd_inner(dataset, backbone, batch_size, device, tau, engine, name)
            out.append(rets)
            torch.cuda.empty_cache()
            
    return out 

    
# C&W Benchmark Function
def test_cw(backbone, test_dataset, device, attack_configs, suffix, batch_size = 64):
    # Reporting Empirical Robustness
    print("Evaluating Empirical Robustness against CW adversary")
    print("Initializing Attack Engines...\n\n")

    attack_engines = []
   
    for name, alg, param in attack_configs: 
        attack_engines.append((name, alg(backbone, device, param)))

    # Display Attack Algorithms
    print("<<< ATTACK ALGORITHMS >>>")
    for name, engine in attack_engines:
        print("Mode: ", name)
        print(engine)
    
    out = []
    for dset_name, dataset in test_dataset:
        print(f"Evalaution on Dataset: {dset_name}")
        # Run Standard Benchmark for each dataset
        feat, mfeat_norm = feat_ext(backbone, dataset, batch_size, device)
        tar, far, acc, tau = benchmark_standard(feat, target_far = 1e-3)
        print(f"TAR: {tar:.4f}\tFAR: {far:.4f}\t ACC: {acc:.4f}\t TAU: {tau:.4f}\tAvg. FN: {mfeat_norm:.4f}")
        
        for name, engine in attack_engines:
            rets = cw_inner(dataset, backbone, batch_size, device, tau, engine, name)
        
            out.append(rets)
            torch.cuda.empty_cache()
    
    return out


# Inner function for evaluating PGD
def pgd_inner(dataset, backbone, batch_size, device, tau, attack_engine, name):
    # Untargeted Attack
    backbone = backbone.to(device)
    data, _ = dataset
    print("Total Attacks: ", name)
    ret_mem = []
    
    for c in name:
        if c == "U":
            print("Untargeted Attack Start...")
            # Untargeted Attack    
            
            unique_dataset = torch.cat(data,dim=0).unique(dim=0)
            n_dset = unique_dataset.size(0)
            print(f"Total # of Unique Images: {n_dset}")

            unt_score = 0
            start = 0
            while start < n_dset:
                end = min(start + batch_size, n_dset)
                blk = (unique_dataset[start:end] - 127.5) / 255
                blk_adv= attack_engine.attack_untargeted(blk.to(device))

                emb = backbone(blk.to(device))
                emb_adv = backbone(blk_adv.to(device))

                rob = (F.cosine_similarity(emb.cpu(), emb_adv.cpu()) > tau)
                unt_score += rob.sum().item()
                start = end


                del blk
                del blk_adv
                del emb
                del emb_adv

            unt_acc = unt_score / n_dset    

            torch.cuda.empty_cache()
            print(f"Untargted ACC: {unt_acc:.4f}")
            ret_mem.append(unt_acc)
        
        elif c == "I":
            print("Impersonation Attack Start...")
            # Targeted Attack
            t_score = 0   
            for dset in data:
                # Only Use False Pairs!
                left, right = dset[::2], dset[1::2]
                left_neg = left.reshape(20, -1, 3, 112, 112)[1::2,:,:,:,:].reshape(-1,3,112,112)
                right_neg = right.reshape(20, -1, 3, 112, 112)[1::2,:,:,:,:].reshape(-1,3,112,112)

                n_nset = left_neg.size(0)
                start = 0

                while start < n_nset:
                    end = min(start + batch_size, n_nset)
                    l_blk = (left_neg[start:end] - 127.5) / 255
                    r_blk = (right_neg[start:end] - 127.5) / 255
                    l_emb = backbone(l_blk.to(device))
                    r_emb = backbone(r_blk.to(device))

                    # Attack on Left
                    l_adv = attack_engine.attack_targeted(l_blk, r_emb, "I")
                    l_adv_emb = backbone(l_adv.to(device))
                    rob1 = F.cosine_similarity(l_adv_emb.cpu(), r_emb.cpu()) < tau

                    # Attack on Right
                    r_adv = attack_engine.attack_targeted(r_blk, l_emb, "I")
                    r_adv_emb = backbone(r_adv.to(device))
                    rob2 = F.cosine_similarity(l_emb.cpu(), r_adv_emb.cpu()) < tau

                    # Merge
                    t_score += (rob1 & rob2).sum().item()


                    del l_emb
                    del r_emb
                    del l_adv
                    del l_adv_emb
                    del r_adv
                    del r_adv_emb

                    start = end

            torch.cuda.empty_cache()
            t_acc = t_score / (n_nset * 2)

            print(f"Impersonation ACC: {t_acc:.4f}.")
            ret_mem.append(t_acc)
            
        elif c == "D":
            
            unt2_score = 0   
            for dset in data:
                # Only Use True Pairs!
                left, right = dset[::2], dset[1::2]
                left_pos = left.reshape(20, -1, 3, 112, 112)[::2,:,:,:,:].reshape(-1,3,112,112)
                right_pos = right.reshape(20, -1, 3, 112, 112)[::2,:,:,:,:].reshape(-1,3,112,112)

                n_pset = left_pos.size(0)
                start = 0

                while start < n_pset:
                    end = min(start + batch_size, n_pset)
                    l_blk = (left_pos[start:end] - 127.5) / 255
                    r_blk = (right_pos[start:end] - 127.5) / 255
                    l_emb = backbone(l_blk.to(device))
                    r_emb = backbone(r_blk.to(device))

                    # Attack on Left
                    l_adv = attack_engine.attack_targeted(l_blk, r_emb, "D")
                    l_adv_emb = backbone(l_adv.to(device))
                    rob1 = F.cosine_similarity(l_adv_emb.cpu(), r_emb.cpu()) > tau

                    # Attack on Right
                    r_adv = attack_engine.attack_targeted(r_blk, l_emb, "D")
                    r_adv_emb = backbone(r_adv.to(device))
                    rob2 = F.cosine_similarity(l_emb.cpu(), r_adv_emb.cpu()) > tau

                    # Merge
                    unt2_score += (rob1 & rob2).sum().item()


                    del l_emb
                    del r_emb
                    del l_adv
                    del l_adv_emb
                    del r_adv
                    del r_adv_emb

                    start = end

            torch.cuda.empty_cache()
            unt2_acc = unt2_score / (n_pset * 2)

            print(f"Dodging ACC: {unt2_acc:.4f}.")
            ret_mem.append(unt2_acc)
        
    
    return ret_mem


# Inner function for evaluating CW
def cw_inner(dataset, backbone, batch_size, device, tau, attack_engine, name):
    backbone = backbone.to(device)
    data, _ = dataset
    ret_mem = [] 
    print("Total Attacks: ", name)
    for c in name:
        if c == "U":
            print("Untargeted Attack Start...")
            # Untargeted Attack    
            
            unique_dataset = torch.cat(data,dim=0).unique(dim=0)
            n_dset = unique_dataset.size(0)
            print(f"Total # of Unique Images: {n_dset}")

            unt_score = 0
            start = 0
            eps_mem = []
            while start < n_dset:
                end = min(start + batch_size, n_dset)
                blk = (unique_dataset[start:end] - 127.5) / 255
                blk_adv= attack_engine.attack_untargeted(blk.to(device))

                emb = backbone(blk.to(device))
                emb_adv = backbone(blk_adv.to(device))
                eps = (blk - blk_adv.cpu()).norm(p=2, dim=[1,2,3]).cpu()
                success = (F.cosine_similarity(emb.cpu(), emb_adv.cpu()) < tau)
                eps_mem.append(eps[success])
                unt_score += (~success).sum().item()
                start = end


                del blk
                del blk_adv
                del emb
                del emb_adv

            unt_acc = unt_score / n_dset    
            eps_mem = torch.cat(eps_mem)
            med_eps = torch.median(eps_mem).item()
            
            torch.cuda.empty_cache()
            print(f"Untargted ACC: {unt_acc:.4f}")
            print(f"Med. eps: {med_eps:.4f}")
            ret_mem.append((unt_acc, med_eps))
        
        elif c == "I":
            print("Impersonation Attack Start...")
            # Targeted Attack
            t_score = 0   
            eps_mem = []
            for dset in data:
                # Only Use False Pairs!
                left, right = dset[::2], dset[1::2]
                left_neg = left.reshape(20, -1, 3, 112, 112)[1::2,:,:,:,:].reshape(-1,3,112,112)
                right_neg = right.reshape(20, -1, 3, 112, 112)[1::2,:,:,:,:].reshape(-1,3,112,112)

                n_nset = left_neg.size(0)
                start = 0

                
                while start < n_nset:
                    end = min(start + batch_size, n_nset)
                    l_blk = (left_neg[start:end] - 127.5) / 255
                    r_blk = (right_neg[start:end] - 127.5) / 255
                    l_emb = backbone(l_blk.to(device))
                    r_emb = backbone(r_blk.to(device))

                    # Attack on Left
                    l_adv = attack_engine.attack_targeted(l_blk, r_emb, "I")
                    l_adv_emb = backbone(l_adv.to(device))
                    success1 = F.cosine_similarity(l_adv_emb.cpu(), r_emb.cpu()) > tau
                    eps1 = (l_blk - l_adv.cpu()).norm(p=2, dim=[1,2,3]).cpu()
                    

                    # Attack on Right
                    r_adv = attack_engine.attack_targeted(r_blk, l_emb, "I")
                    r_adv_emb = backbone(r_adv.to(device))
                    success2 = F.cosine_similarity(l_emb.cpu(), r_adv_emb.cpu()) > tau
                    eps2 = (r_blk - r_adv.cpu()).norm(p=2, dim=[1,2,3]).cpu()
                    

                    # Merge
                    t_score += (~success1 & ~success2).sum().item()
                    
                    eps_mem.append(eps1[success1])
                    eps_mem.append(eps2[success2])

                    del l_emb
                    del r_emb
                    del l_adv
                    del l_adv_emb
                    del r_adv
                    del r_adv_emb

                    start = end

            torch.cuda.empty_cache()
            eps_mem = torch.cat(eps_mem)
            med_eps = torch.median(eps_mem).item()            
            t_acc = t_score / (n_nset * 2)

            print(f"Impersonation ACC: {t_acc:.4f}.")
            print(f"Med. eps: {med_eps:.4f}")       
            ret_mem.append((t_acc, med_eps))
            
        elif c == "D":
            
            unt2_score = 0   
            eps_mem = []            
            for dset in data:
                # Only Use True Pairs!
                left, right = dset[::2], dset[1::2]
                left_pos = left.reshape(20, -1, 3, 112, 112)[::2,:,:,:,:].reshape(-1,3,112,112)
                right_pos = right.reshape(20, -1, 3, 112, 112)[::2,:,:,:,:].reshape(-1,3,112,112)

                n_pset = left_pos.size(0)
                start = 0

                while start < n_pset:
                    end = min(start + batch_size, n_pset)
                    l_blk = (left_pos[start:end] - 127.5) / 255
                    r_blk = (right_pos[start:end] - 127.5) / 255
                    l_emb = backbone(l_blk.to(device))
                    r_emb = backbone(r_blk.to(device))

                    # Attack on Left
                    l_adv = attack_engine.attack_targeted(l_blk, r_emb, "D")
                    l_adv_emb = backbone(l_adv.to(device))
                    success1 = F.cosine_similarity(l_adv_emb.cpu(), r_emb.cpu()) < tau
                    eps1 = (l_blk - l_adv.cpu()).norm(p=2, dim=[1,2,3]).cpu()
                    
                    # Attack on Right
                    r_adv = attack_engine.attack_targeted(r_blk, l_emb, "D")
                    r_adv_emb = backbone(r_adv.to(device))
                    success2 = F.cosine_similarity(l_emb.cpu(), r_adv_emb.cpu()) < tau
                    eps2 = (r_blk - r_adv.cpu()).norm(p=2, dim=[1,2,3]).cpu()

                    # Merge
                    unt2_score += (~success1 & ~success1).sum().item()

                    eps_mem.append(eps1[success1])
                    eps_mem.append(eps2[success2])
                    
                    del l_emb
                    del r_emb
                    del l_adv
                    del l_adv_emb
                    del r_adv
                    del r_adv_emb

                    start = end

            torch.cuda.empty_cache()
            eps_mem = torch.cat(eps_mem)
            med_eps = torch.median(eps_mem).item()             
            unt2_acc = unt2_score / (n_pset * 2)

            print(f"Dodging ACC: {unt2_acc:.4f}.")
            print(f"Med. eps: {med_eps:.4f}")                   
            ret_mem.append(unt2_acc)
        
    
    return ret_mem
