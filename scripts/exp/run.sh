#!/usr/bin/env bash
# Run covmatch-style distillation with hyperbolic lift, range/null subspaces, and base hyperbolic InfoNCE.
# Usage: bash scripts/covmatch_style_ours/run.sh <gpu_id> <run_name>
# Example: bash scripts/covmatch_style_ours/run.sh 0 hyp_rn_flickr8k_100

set -e
cd "$(dirname "$0")/../.."

NUM_QUERIES=${3:-100}

CUDA_VISIBLE_DEVICES=${1:-0} python main.py \
    --log_freq 10 \
    --dataset flickr8k \
    --name "${2:-covmatch_style_ours_hyp}" \
    --num_queries ${NUM_QUERIES} \
    --batch_size_train 64 \
    --batch_size_test 64 \
    --lr_txt 1 \
    --lr_img 1 \
    --Iteration 3000 \
    --epoch_eval_train 100 \
    --eval_it 10 \
    --num_eval 5 \
    --wandb\
    --seed 0 \
    --rho 2 \
    --lamda 0.0 \
    --w_cov 0.0 \
    --w_hyp 1.0 \
    --hyperbolic_c 1.0 \
    --hyp_scale 1.0 \
    --hyp_temperature 0.07 \
    --w_hyp_nce 1.0 \
    --w_range_xcov 0.8 \
    --w_null_xcov 0.4 \
    --w_null_compress 0.1 \
    --relevance_temp 0.07 \
    --rn_energy 0.95 \
    --rn_max_rank -1 \
    --rn_ot_reg 0.05 \
    --rn_ot_iters 20 \
    --use_wbce False