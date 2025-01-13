#!/bin/bash

set -e
set -x


CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port 12345 --nproc_per_node=2 \
    main_pretrain_sfa.py \
    --dataset COCO \
    --data-dir "/media/mmlab206/18b85164-d1ff-4800-80c0-08899eb52cae1/yonghonglin/coco2017" \
    --output-dir "./output/sfa_v4_sigma0.5_slotcon_coco_r50_800epbs256" \
    \
    --arch resnet50 \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 256 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 1.0 \
    \
    --batch-size 256 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 800 \
    --fp16 \
    \
    --print-freq 1 \
    --save-freq 1 \
    --auto-resume \
    --num-workers 8