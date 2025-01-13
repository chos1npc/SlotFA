import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import numpy as np
import matplotlib.pyplot as plt


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
    
    def feature_std_loss(self, features):
        """
        計算每張圖像在每個位置上的通道方差損失，保持空間資訊
        
        :param features: 來自backbone與projector的特徵，形狀為 [b, channel, h, w]
        :return: 方差損失
        """
        # Calculate mean across channels for each spatial position
        mean_features = features.mean(dim=1, keepdim=True)  # shape: [b, 1, h, w]
        
        # Center the features by subtracting the mean
        centered_features = features - mean_features  # shape: [b, channel, h, w]
        
        # Calculate variance across channels for each spatial position
        variances = centered_features.var(dim=1, unbiased=False)  # shape: [b, h, w]
        
        # Calculate standard deviation
        std = torch.sqrt(variances + 1e-4)  # shape: [b, h, w]
        
        # Calculate standard deviation loss
        std_loss = torch.mean(F.relu(1 - std))  # scalar value
        
        return std_loss
    
    def per_img_std_loss(self, centers):
        """
        計算每張圖像上slot centers的方差損失
        
        :param centers: slot 中心 [batch, num_slots, dim]
        :return: 方差損失
        """

        '''
        
        '''
        # 檢查哪些 slots 被使用，計算有效 slot 的掩碼
        valid_slots = (centers.sum(dim=2) != 0)  # [batch, num_slots]
        
        # 計算有效 slots 的數量
        valid_slot_counts = valid_slots.sum(dim=1)  # [batch]
        
        # 過濾掉有效 slot 少於 2 的批次（這些批次不計算方差）
        valid_batch_mask = valid_slot_counts > 1  # [batch]
        valid_centers = centers[valid_batch_mask]  # [batch', num_slots, dim]
        valid_slots = valid_slots[valid_batch_mask]  # [batch', num_slots]
        
        # 計算每個批次中有效 slots 的平均值
        mean_centers = torch.sum(valid_centers * valid_slots.unsqueeze(-1), dim=1) / valid_slot_counts[valid_batch_mask].unsqueeze(-1)  # [batch', dim]
        
        # 減去平均值以計算方差
        centered_centers = valid_centers - mean_centers.unsqueeze(1)  # [batch', num_slots, dim]
        
        # 計算有效 slots 的方差
        variances = torch.var(centered_centers, dim=1, unbiased=False)  # [batch', dim]
        std = torch.sqrt(variances + 1e-4)  # [batch', dim]
        # 計算方差損失
        std_loss = torch.mean(F.relu(1 - std), dim=1)  # [batch']
        
        # 返回 batch 平均方差損失
        return std_loss.mean() if valid_batch_mask.sum() > 0 else torch.tensor(0.0, device=centers.device)

    def forward(self, input):
        crops, coords, flags = input
        x1, x2 = self.projector_q(self.encoder_q(crops[0])), self.projector_q(self.encoder_q(crops[1]))
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            y1, y2 = self.projector_k(self.encoder_k(crops[0])), self.projector_k(self.encoder_k(crops[1]))
            
        (q1, score_q1), (q2, score_q2) = self.grouping_q(x1), self.grouping_q(x2)
        q1_aligned, q2_aligned = self.invaug(score_q1, coords[0], flags[0]), self.invaug(score_q2, coords[1], flags[1])
        with torch.no_grad():
            (k1, score_k1), (k2, score_k2) = self.grouping_k(y1), self.grouping_k(y2)
            k1_aligned, k2_aligned = self.invaug(score_k1, coords[0], flags[0]), self.invaug(score_k2, coords[1], flags[1])
                
        loss = self.group_loss_weight * self.self_distill(q1_aligned.permute(0, 2, 3, 1).flatten(0, 2), k2_aligned.permute(0, 2, 3, 1).flatten(0, 2)) \
             + self.group_loss_weight * self.self_distill(q2_aligned.permute(0, 2, 3, 1).flatten(0, 2), k1_aligned.permute(0, 2, 3, 1).flatten(0, 2))

        self.update_center(torch.cat([score_k1, score_k2]).permute(0, 2, 3, 1).flatten(0, 2))

        # loss += (1. - self.group_loss_weight) * self.ctr_loss_filtered(q1, k2, score_q1, score_k2) \
        #       + (1. - self.group_loss_weight) * self.ctr_loss_filtered(q2, k1, score_q2, score_k1)
        
        return loss
    
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