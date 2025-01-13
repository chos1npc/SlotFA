#!/bin/bash

set -e
set -x


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 12345 --nproc_per_node=4 \
    main_pretrain_sfa.py \
    --dataset COCO \
    --data-dir "./coco2017" \
    --output-dir "./output/SlotFA_coco_r50_800epbs512" \
    \
    --arch resnet50 \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 256 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 1.0 \
    \
    --batch-size 512 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 800 \
    --fp16 \
    \
    --print-freq 1 \
    --save-freq 10 \
    --auto-resume \
    --num-workers 8