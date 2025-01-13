import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random

class DINOHead(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class DINOHead2d(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Conv2d(in_dim, bottleneck_dim, 1)
        else:
            layers = [nn.Conv2d(in_dim, hidden_dim, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Conv2d(hidden_dim, bottleneck_dim, 1))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x
    
class SimpleFA(nn.Module):
    def __init__(self, sigma1=1, sigma2=1):
        super(SimpleFA, self).__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2
   
    def forward(self, x, slot_assign):
        # 為每個slot生成高斯噪聲
        slot_noises = self.generate_slot_noises(slot_assign)

        # 同步slot噪聲到所有GPU上
        for slot in slot_noises:
            slot_noises[slot] = self.sync_across_gpus(slot_noises[slot])
        # 應用噪聲到x上
        x = self.apply_slot_noises(x, slot_assign, slot_noises)
        # print("x shape:", x.shape)
        # x = x+noise
        return x

    def img_forward(self, x):
        
        x = self.apply_noises(x)
        return x
    
    def generate_noises(self, device):
        alpha = torch.normal(mean=1.0, std=self.sigma1, size=(256, 7, 7), device=device)
        beta = torch.normal(mean=0.0, std=self.sigma2, size=(256, 7, 7), device=device)
        return alpha, beta

    def generate_slot_noises(self, slot_assign):
        # print(unique_slots)
        slot_noises = {}
        for slot in range(256):
            alpha = torch.normal(mean=1.0, std=self.sigma1, size=(256,), device=slot_assign.device)
            beta = torch.normal(mean=0.0, std=self.sigma2, size=(256,), device=slot_assign.device)
            slot_noises[slot] = (alpha, beta)

        return slot_noises

    def sync_across_gpus(self, tensor_tuple):
        alpha, beta = tensor_tuple
        dist.broadcast(alpha, src=0)
        dist.broadcast(beta, src=0)
        return alpha, beta

    def apply_noises(self, x):
        batch_size, channels, height, width = x.shape
        for i in range(batch_size):
            alpha, beta = self.generate_noises(device=x.device)
            x[i] = alpha * x[i] + beta
        return x
    
    def apply_slot_noises(self, x, slot_assign, slot_noises):
        batch_size, channels, height, width = x.shape
        x_aug = torch.zeros_like(x)
        for i in range(batch_size):
            for j in range(height):
                for k in range(width):
                    slot = slot_assign[i, j, k].item()
                    alpha, beta = slot_noises[slot]
                    x_aug[i, :, j, k] = alpha * x[i, :, j, k] + beta
        x = x_aug
        return x

class SimpleFA_SelectAug(nn.Module):
    def __init__(self, sigma1=0.5, sigma2=0.5):
        super(SimpleFA_SelectAug, self).__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def forward(self, x, slot_assign):
        # 獲取批次中實際被分配到的槽
        assigned_slots = self.get_assigned_slots(slot_assign)

        # 在已分配的槽中隨機選擇 50% 進行增強
        augment_slots = self.select_slots_to_augment(assigned_slots)

        # 為所有槽生成增強參數（增強槽和非增強槽）
        slot_noises = self.generate_slot_noises(augment_slots, x.device)

        # 同步槽的增強參數到所有 GPU 上
        # for slot in slot_noises:
        #     slot_noises[slot] = self.sync_across_gpus(slot_noises[slot])

        # 應用增強到特徵（使用向量化操作）
        x = self.apply_slot_noises(x, slot_assign, slot_noises)
        return x

    def get_assigned_slots(self, slot_assign):
        assigned_slots = torch.unique(slot_assign)
        return assigned_slots.tolist()

    def select_slots_to_augment(self, assigned_slots):
        num_slots = len(assigned_slots)
        num_augment = int(num_slots * 0.75)
        augment_slots = random.sample(assigned_slots, num_augment)
        return augment_slots

    def generate_slot_noises(self, augment_slots, device):
        slot_noises = {}
        for slot in range(256):
            if slot in augment_slots:
                alpha = torch.normal(mean=1.0, std=self.sigma1, size=(256,), device=device)
                beta = torch.normal(mean=0.0, std=self.sigma2, size=(256,), device=device)
            else:
                alpha = torch.ones(256, device=device)
                beta = torch.zeros(256, device=device)
            slot_noises[slot] = (alpha, beta)
        return slot_noises

    def sync_across_gpus(self, tensor_tuple):
        alpha, beta = tensor_tuple
        dist.broadcast(alpha, src=0)
        dist.broadcast(beta, src=0)
        return alpha, beta

    def apply_slot_noises(self, x, slot_assign, slot_noises):
        batch_size, channels, height, width = x.shape
        # 將 x 攤平成 (batch_size * height * width, channels)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, channels)
        # 將 slot_assign 攤平成 (batch_size * height * width,)
        slot_assign_flat = slot_assign.reshape(-1)
        # 創建 alpha 和 beta 矩陣
        alpha = torch.stack([slot_noises[slot][0] for slot in range(256)], dim=0)
        beta = torch.stack([slot_noises[slot][1] for slot in range(256)], dim=0)
        # 根據 slot_assign_flat 獲取對應的 alpha 和 beta
        alpha_assigned = alpha[slot_assign_flat]
        beta_assigned = beta[slot_assign_flat]
        # 應用增強
        x_aug_flat = alpha_assigned * x_flat + beta_assigned
        # 恢復 x_aug 的形狀
        x_aug = x_aug_flat.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return x_aug

class SemanticGrouping(nn.Module):
    def __init__(self, num_slots, dim_slot, temp=0.07, eps=1e-6):
        super().__init__()
        self.num_slots = num_slots
        self.dim_slot = dim_slot
        self.temp = temp
        self.eps = eps

        self.slot_embed = nn.Embedding(num_slots, dim_slot)

    def forward(self, x):
        x_prev = x
        slots = self.slot_embed(torch.arange(0, self.num_slots, device=x.device)).unsqueeze(0).repeat(x.size(0), 1, 1)
        # dots means the similarity between slots and features 
        dots = torch.einsum('bkd,bdhw->bkhw', F.normalize(slots, dim=2), F.normalize(x, dim=1))
        # attn means the attention map between slots and features
        attn = (dots / self.temp).softmax(dim=1) + self.eps
        # slots means the weighted sum of features
        slots = torch.einsum('bdhw,bkhw->bkd', x_prev, attn / attn.sum(dim=(2, 3), keepdim=True))
     
        return slots, dots

class SlotCon(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out
        self.teacher_momentum = args.teacher_momentum

        self.num_channels = 512 if args.arch in ('resnet18', 'resnet34') else 2048
        self.encoder_q = encoder(head_type='early_return')
        self.encoder_k = encoder(head_type='early_return')

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        self.group_loss_weight = args.group_loss_weight
        self.student_temp = args.student_temp
        self.teacher_temp = args.teacher_temp
            
        self.projector_q = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        self.projector_k = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_q)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)

        self.num_prototypes = args.num_prototypes
        self.center_momentum = args.center_momentum
        self.register_buffer("center", torch.zeros(1, self.num_prototypes))
        self.grouping_q = SemanticGrouping(self.num_prototypes, self.dim_out, self.teacher_temp)
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out, self.teacher_temp)
        self.predictor_slot = DINOHead(self.dim_out, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        self.simple_fa = SimpleFA_SelectAug()
        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor_slot)
            
        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.K = int(args.num_instances * 1. / args.world_size / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    def re_init(self, args):
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        self.k += 1
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)  

    def invaug(self, x, coords, flags):
        N, C, H, W = x.shape

        batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
        coords_rescaled = coords.clone()
        coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
        coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
        coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
        coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2
        coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)
        
        x_aligned = torchvision.ops.roi_align(x, coords_with_idxs, (H, W), aligned=True)
        x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])
        return x_flipped

    def self_distill(self, q, k):
        q = F.log_softmax(q / self.student_temp, dim=-1)
        k = F.softmax((k - self.center) / self.teacher_temp, dim=-1)
        return torch.sum(-k * q, dim=-1).mean()


    def ctr_loss_filtered(self, q, k, score_q, score_k, tau=0.2):
        # print("before flatten q shape:", q.shape, "before flatten k shape:", k.shape)
        # before flatten q shape: torch.Size([128, 256, 256]) before flatten k shape: torch.Size([128, 256, 256]) batch_size, num_slots, dim_slot
        q = q.flatten(0, 1)
        k = F.normalize(k.flatten(0, 1), dim=1)
        # print("after flatten q shape:", q.shape, "after flatten k shape:", k.shape)
        # after flatten q shape: torch.Size([32768, 256]) after flatten k shape: torch.Size([32768, 256])
        # print("attention map q shape:", score_q.shape, "attention map k shape:", score_k.shape)   #attention map q shape: torch.Size([128, 256, 7, 7]) attention map k shape: torch.Size([128, 256, 7, 7])
        mask_q = (torch.zeros_like(score_q).scatter_(1, score_q.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()    #計算出每個slot的mask 
        mask_k = (torch.zeros_like(score_k).scatter_(1, score_k.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        # print("mask q shape:", mask_q.shape, "mask k shape:", mask_k.shape) #mask q shape: torch.Size([128, 256]) mask k shape: torch.Size([128, 256])
        mask_intersection = (mask_q * mask_k).view(-1)  #計算出兩視圖共有的slot
        # print("original mask intersection shape:", (mask_q*mask_k).shape)
        # print("mask intersection shape:", mask_intersection.shape)  #mask intersection shape: torch.Size([32768])
        idxs_q = mask_intersection.nonzero().squeeze(-1)    #取出共有的slot的index
        # print("idxs q shape:", idxs_q.shape)

        mask_k = concat_all_gather(mask_k.view(-1)) #將mask_k的值擴展到所有GPU
        # print("mask k shape:", mask_k.shape) #mask k shape: torch.Size([65536])
        idxs_k = mask_k.nonzero().squeeze(-1)   #取出所有slot的index
        # print("idxs k shape:", idxs_k.shape)
        N = k.shape[0]
        logits = torch.einsum('nc,mc->nm', [F.normalize(self.predictor_slot(q[idxs_q]), dim=1), concat_all_gather(k)[idxs_k]]) / tau    #計算出共有的slot的logits
        labels = mask_k.cumsum(0)[idxs_q + N * torch.distributed.get_rank()] - 1    #計算出共有的slot的label
        # print("logits shape:", logits.shape, "labels shape:", labels.shape) #logits shape: torch.Size([1031, 7403]) labels shape: torch.Size([1031])
        return F.cross_entropy(logits, labels) * (2 * tau)
    
    def aug_ctr_loss_filtered(self, q, k, score_q, score_k, tau=0.2):
        # print("before flatten q shape:", q.shape, "before flatten k shape:", k.shape)
        # before flatten q shape: torch.Size([128, 256, 256]) before flatten k shape: torch.Size([128, 256, 256]) batch_size, num_slots, dim_slot
        q = q.flatten(0, 1)
        k = F.normalize(k.flatten(0, 1), dim=1)
        # print("after flatten q shape:", q.shape, "after flatten k shape:", k.shape)
        # after flatten q shape: torch.Size([32768, 256]) after flatten k shape: torch.Size([32768, 256])
        # print("attention map q shape:", score_q.shape, "attention map k shape:", score_k.shape)   #attention map q shape: torch.Size([128, 256, 7, 7]) attention map k shape: torch.Size([128, 256, 7, 7])
        mask_q = (torch.zeros_like(score_q).scatter_(1, score_q.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()    #計算出每個slot的mask 
        mask_k = (torch.zeros_like(score_k).scatter_(1, score_k.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        # print("mask q shape:", mask_q.shape, "mask k shape:", mask_k.shape) #mask q shape: torch.Size([128, 256]) mask k shape: torch.Size([128, 256])
        mask_intersection = (mask_q * mask_k).view(-1)  #計算出兩視圖共有的slot
        # print("original mask intersection shape:", (mask_q*mask_k).shape)
        # print("mask intersection shape:", mask_intersection.shape)  #mask intersection shape: torch.Size([32768])
        idxs_q = mask_intersection.nonzero().squeeze(-1)    #取出共有的slot的index
        # print("idxs q shape:", idxs_q.shape)

        mask_k = concat_all_gather(mask_k.view(-1)) #將mask_k的值擴展到所有GPU
        # print("mask k shape:", mask_k.shape) #mask k shape: torch.Size([65536])
        idxs_k = mask_k.nonzero().squeeze(-1)   #取出所有slot的index
        # print("idxs k shape:", idxs_k.shape)
        N = k.shape[0]
        with torch.no_grad():
            logits = torch.einsum('nc,mc->nm', [F.normalize(self.predictor_slot(q[idxs_q]), dim=1), concat_all_gather(k)[idxs_k]]) / tau    #計算出共有的slot的logits
        labels = mask_k.cumsum(0)[idxs_q + N * torch.distributed.get_rank()] - 1    #計算出共有的slot的label
        # print("logits shape:", logits.shape, "labels shape:", labels.shape) #logits shape: torch.Size([1031, 7403]) labels shape: torch.Size([1031])
        return F.cross_entropy(logits, labels) * (2 * tau) 

    def forward(self, input, epoch):
        crops, coords, flags = input
        #sfa_v4 simple augument on proj and alignment both teacher,student and teacher,aug student 
        x1, x2 = self.projector_q(self.encoder_q(crops[0], self.grouping_q, self.projector_q)), self.projector_q(self.encoder_q(crops[1], self.grouping_q, self.projector_q))
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            y1, y2 = self.projector_k(self.encoder_k(crops[0])), self.projector_k(self.encoder_k(crops[1]))

        (q1, score_q1), (q2, score_q2) = self.grouping_q(x1), self.grouping_q(x2)
        q1_aligned, q2_aligned = self.invaug(score_q1, coords[0], flags[0]), self.invaug(score_q2, coords[1], flags[1])

        with torch.no_grad():
            (k1, score_k1), (k2, score_k2) = self.grouping_k(y1), self.grouping_k(y2)
            k1_aligned, k2_aligned = self.invaug(score_k1, coords[0], flags[0]), self.invaug(score_k2, coords[1], flags[1])
        
        # augmentations for the key encoder
        aug_x1, aug_x2 = self.simple_fa(x1, score_q1.argmax(dim=1)), self.simple_fa(x2, score_q2.argmax(dim=1))
        with torch.no_grad():
            (aug_q1, score_aug_q1), (aug_q2, score_aug_q2) = self.grouping_q(aug_x1), self.grouping_q(aug_x2)
            aug_q1_aligned, aug_q2_aligned = self.invaug(score_aug_q1, coords[0], flags[0]), self.invaug(score_aug_q2, coords[1], flags[1])
        
        ori_loss = self.group_loss_weight * self.self_distill(q1_aligned.permute(0, 2, 3, 1).flatten(0, 2), k2_aligned.permute(0, 2, 3, 1).flatten(0, 2)) \
             + self.group_loss_weight * self.self_distill(q2_aligned.permute(0, 2, 3, 1).flatten(0, 2), k1_aligned.permute(0, 2, 3, 1).flatten(0, 2))
        aug_loss = self.group_loss_weight * self.self_distill(aug_q1_aligned.permute(0, 2, 3, 1).flatten(0, 2), k2_aligned.permute(0, 2, 3, 1).flatten(0, 2)) \
             + self.group_loss_weight * self.self_distill(aug_q2_aligned.permute(0, 2, 3, 1).flatten(0, 2), k1_aligned.permute(0, 2, 3, 1).flatten(0, 2))
        
        self.update_center(torch.cat([score_k1, score_k2]).permute(0, 2, 3, 1).flatten(0, 2))

        
        loss = ori_loss + aug_loss

        return ori_loss, aug_loss, loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class SlotConEval(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out

        self.num_channels = 512 if args.arch in ('resnet18', 'resnet34') else 2048
        self.encoder_k = encoder(head_type='early_return')
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        self.projector_k = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        for param_k in self.projector_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        self.num_prototypes = args.num_prototypes
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out)
        for param_k in self.grouping_k.parameters():
            param_k.requires_grad = False  # not update by gradient

    def forward(self, x):
        with torch.no_grad():
            slots, probs = self.grouping_k(self.projector_k(self.encoder_k(x)))
            return slots, probs