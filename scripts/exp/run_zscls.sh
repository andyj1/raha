#!/usr/bin/env bash
# Run zero-shot classification distillation.
# Usage: bash scripts/exp/run_zscls.sh <gpu_id> <run_name> [zscls_dataset]
# Example: bash scripts/exp/run_zscls.sh 0 zscls_imagenet imagenet

set -e
cd "$(dirname "$0")/../.."

ZSCLS_DS=${3:-cifar100}

export MDD_BASE_DIR="${MDD_BASE_DIR:-/mnt/sdc/mdd_datasets}"
export ZSCLS_BASE_DIR="${ZSCLS_BASE_DIR:-/mnt/sdk/adv_datasets}"

CUDA_VISIBLE_DEVICES=${1:-0} python main_zscls.py \
    --zscls_dataset "${ZSCLS_DS}" \
    --name "${2:-covmatch_style_ours_zscls}" \
    --num_queries 100 \
    --batch_size_train 64 \
    --batch_size_test 64 \
    --lr_txt 1 \
    --lr_img 1 \
    --Iteration 3000 \
    --epoch_eval_train 100 \
    --eval_it 10 \
    --num_eval 5 \
    --wandb \
    --seed 0 \
    --rho 2 \
    --lamda 0.0 \
    --w_cov 0.0
