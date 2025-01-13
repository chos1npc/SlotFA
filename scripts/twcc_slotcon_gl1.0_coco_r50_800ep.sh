#!/bin/bash

#SBATCH --job-name=slotcon_coco_r50_800ep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --account=MST111409
#SBATCH --partition=gp4d


module load miniconda3
conda activate yonghonglin

set -e
set -x


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 12348 --nproc_per_node=4 \
    main_pretrain.py \
    --dataset COCO \
    --data-dir "/home/mmlab206/coco2017" \
    --output-dir "./output/slotcon_gl1.0_coco_r50_800ep" \
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
    --save-freq 20 \
    --auto-resume \
    --num-workers 8
