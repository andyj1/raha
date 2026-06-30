# Rank-Aware Hyperbolic Alignment for Vision-Language Dataset Distillation (ECCV 2026)

Official implementation of *Rank-Aware Hyperbolic Alignment for Vision-Language Dataset Distillation* (RAHA), a geometry-aware method for condensing large vision-language datasets into compact synthetic sets while preserving cross-modal retrieval structure.

[Paper](https://arxiv.org/abs/2606.29464) | [Project Page](https://andyj1.github.io/raha)

## News

- Will be updated soon!

## About

RAHA distills compact image-text datasets by combining **rank-aware distribution matching** with **hyperbolic image-text alignment**. It lifts multimodal representations to hyperbolic space, estimates an adaptive rank-k shared semantic range from real batch coupling, matches relevance in that dominant range, and regularizes the residual subspace so weak correlations do not dominate under tight budgets.

The distillation objective is:

```text
L_total = L_hITC + λ_range L_range + λ_residual L_residual
```

## Dataset

```text
data/
├── datasets/
│   ├── Flickr30k/
│   ├── Flickr8k/
│   └── COCO/
└── annotations/
    ├── flickr30k/
    ├── flickr8k/
    └── coco/
```

Defaults in `distill_raha.py` follow the Karpathy-style retrieval splits used in the paper:

image roots such as `./data/datasets/Flickr30k/` and annotation root `./data/annotations/` when using the `flickr` / `flickr8k` / `coco` options.

Each distilled query contains one `3×224×224` synthetic image optimized in pixel space and one continuous `768`-dimensional text embedding optimized in embedding space.

## Training

```bash
export CKPT_PATH=/path/to/distilled.pt
./sh/distill.sh <gpu_id> [run_name]
```

Example command (Flickr8k, 500 pairs; matches the paper's default optimization settings in Table A2/A3):

```bash
CUDA_VISIBLE_DEVICES=0 python distill_raha.py \
    --dataset flickr8k \
    --num_queries 500 \
    --batch_size_train 64 \
    --batch_size_test 64 \
    --lr_img 1.0 \
    --lr_txt 1.0 \
    --Iteration 200 \
    --outer_loop 50 \
    --inner_loop 1 \
    --epoch_eval_train 100 \
    --num_eval 5 \
    --curvature 1.0 \
    --lift_scale 1.0 \
    --tau_hitc 0.07 \
    --tau_relevance 0.07 \
    --rank_energy 0.95 \
    --sinkhorn_eps 0.05 \
    --sinkhorn_iters 20 \
    --lambda_range 0.8 \
    --lambda_residual 0.4 \
    --lambda_comp 0.1 \
    --log_dir logs \
    --eval_eval_freq 50 \
    --save_it 50 \
    --seed 1
```

## Evaluation

After distillation, train a retrieval model from scratch on the synthetic set for 100 epochs and evaluate on the real test split:

```bash
export CKPT_PATH=/path/to/distilled.pt
./sh/eval.sh <gpu_id>
```

Example command:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --dataset flickr8k \
    --num_queries 500 \
    --num_eval 5 \
    --epoch_eval_train 100 \
    --ckpt_path /path/to/distilled.pt \
    --image_encoder nfnet \
    --text_encoder bert \
    --batch_size_train 64 \
    --batch_size_test 64
```


## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{jeong2026raha,
  title     = {Rank-Aware Hyperbolic Alignment for Vision-Language Dataset Distillation},
  author    = {Jeong, Jongoh and Lee, Sun-Kyung and Yoon, Kuk-Jin},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
