# **Rank-Aware Hyperbolic Alignment for Vision-Language Dataset Distillation (ECCV 2026)**

> [Jongoh Jeong](https://andyj1.github.io/)1, [Sun-Kyung Lee](https://sites.google.com/view/sklee2014)2, [Kuk-Jin Yoon](https://sites.google.com/site/kjyoon/)1  
> 1Korea Advanced Institute of Science and Technology (KAIST), 2Electronics and Telecommunications Research Institute (ETRI)

---

Official implementation of Rank-Aware Hyperbolic Alignment for Vision-Language Dataset Distillation, a method for condensing a large vision-language dataset into smaller synthetic sets while preserving its downstream performance.

`[arXiv](https://arxiv.org/abs/2606.29464)` | `[Project Page](https://andyj1.github.io/raha)` | `[BibTeX](#citation)`

Main

## About

RAHA distills compact image-text datasets by combining rank-aware distribution matching with hyperbolic image-text alignment. It lifts multimodal representations to hyperbolic space, estimates an adaptive rank-k shared semantic range from real batch coupling, matches relevance in that dominant range, and regularizes the residual subspace so weak correlations do not dominate under tight budgets.

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

- Image roots such as `./data/datasets/Flickr30k/`
- Annotation root `./data/annotations/`
- Dataset options: `flickr`, `flickr8k`, and `coco`

Each distilled query contains one `3×224×224` synthetic image optimized in pixel space and one continuous `768`-dimensional text embedding optimized in embedding space.

## Training

```bash
export CKPT_PATH=/path/to/distilled.pt
./sh/distill.sh <gpu_id> [run_name]
```

## Evaluation

After distillation, train a retrieval model from scratch on the synthetic set for 100 epochs and evaluate on the real test split:

```bash
export CKPT_PATH=/path/to/distilled.pt
./sh/eval.sh <gpu_id>
```

## Citation

If you make a reference to our paper in your research, please cite:

```bibtex
@inproceedings{jeong2026raha,
  title     = {Rank-Aware Hyperbolic Alignment for Vision-Language Dataset Distillation},
  author    = {Jeong, Jongoh and Lee, Sun-Kyung and Yoon, Kuk-Jin},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

