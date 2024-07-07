from dataset import MXFaceDataset
from lr_scheduler import PolynomialLRWarmup
from torch.utils.data import DataLoader
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from torch.utils.tensorboard import SummaryWriter
import logging
import os


from backbones.sllnet import SLLNet
from headers import ECosFace


import torch
from torch import nn
import torch.nn.functional as F

# Many parts of this function was forked from InsightFace: https://github.com/deepinsight/insightface
def train(cfg):
    rank = 0
    wandb_logger = None
    os.makedirs(cfg.output, exist_ok = True)
    init_logging(rank, cfg.output)
    
    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )    
    
    dataset = MXFaceDataset(cfg.rec, 0)
    dataloader = DataLoader(
        dataset, cfg.batch_size, shuffle = True,
        num_workers = cfg.num_workers, pin_memory = True, drop_last = True
    )
    
    backbone = SLLNet([3,4,6,3], emb_size =cfg.emb_size,  fp16 = cfg.fp16)
    header = ECosFace(cfg.num_classes, cfg.emb_size, s = 64, m = 0.35)

    backbone.to("cuda:0").train()
    header.to("cuda:0").train()
    
    cfg.warmup_step = cfg.num_image // cfg.batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.batch_size * cfg.num_epoch
    
    # Note. The behavior of AdamW with weight_decay = 0 is equivalent to usual Adam.
    opt = torch.optim.AdamW(
        [{"params":backbone.parameters()}, {"params": header.parameters()}],
        lr = cfg.lr,
        weight_decay = cfg.weight_decay
    )
    
    sched = PolynomialLRWarmup(
        optimizer = opt, warmup_iters = cfg.warmup_step,
        total_iters = cfg.total_step
    )
    
    global_step = 0
    
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )
    
    
    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    
    loss_fn = nn.CrossEntropyLoss()

    lambd = cfg.lambd
    xi = cfg.xi
    for epoch in range(cfg.num_epoch):
        for img, labels in dataloader:
            img = img.to("cuda:0"); labels = labels.to("cuda:0")
            global_step += 1

            embs = backbone(img)
            logits = header(embs, labels)
            loss = loss_fn(logits, labels)
            
            # Proposed Regularization Loss
            if xi > 0:
                reg_loss = lambd * F.relu(-xi * torch.log(embs.norm(2, dim=1) / xi)).mean()
            else:
                reg_loss = torch.zeros(1, device = "cuda:0").mean()
            loss += reg_loss
        
            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
                opt.zero_grad()
            else:
                loss.backward()
                opt.step()
                opt.zero_grad()

            sched.step()
            

            with torch.no_grad():
                feat_norm = embs.norm(2, dim=1).mean().item()
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, sched.get_last_lr()[0], amp, feat_norm, reg_loss.item())

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if epoch % 5 == 0:
            path_backbone = os.path.join(cfg.output, "model_%d.pt"%epoch)
            path_header = os.path.join(cfg.output, "header_%d.pt"%epoch)
            torch.save(backbone.state_dict(), path_backbone)
            torch.save(header.state_dict(), path_header)
            
    path_backbone = os.path.join(cfg.output, "model_final.pt")
    path_header = os.path.join(cfg.output, "header_final.pt")

    torch.save(backbone.state_dict(), path_backbone)
    torch.save(header.state_dict(), path_header)
    
if __name__ == "__main__":
    from config import config
    import torch
    torch.backends.cudnn.benchmark = True
    train(config)
    