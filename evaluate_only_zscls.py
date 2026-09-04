"""
Evaluate a distilled (image, text, mask) checkpoint on zero-shot classification
for a given dataset. Loads checkpoint, runs evaluate_synset_zscls, reports Top-1, Top-5, ECE.
"""
import argparse
import copy
import os

import numpy as np
import torch

from data import get_zscls_dataloaders, ZSCLS_DATASETS
from main_zscls import print_results
from src.epoch import evaluate_synset_zscls
from src.networks import CLIPModel_full


def main(args):
    if args.dataset not in ZSCLS_DATASETS:
        raise ValueError(
            f"Dataset must be one of {sorted(ZSCLS_DATASETS)} for zscls evaluation. Got: {args.dataset}"
        )

    trainloader, testloader, train_dataset, test_dataset = get_zscls_dataloaders(args)
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    print("Hyper-parameters:\n", args.__dict__)


    student_net = CLIPModel_full(args).to(args.device)
    student_net.eval()
    image_encoder_weights = copy.deepcopy(student_net.image_encoder.state_dict())
    text_encoder_weights = copy.deepcopy(student_net.text_encoder.state_dict())
    del student_net


    ckpt_path = args.ckpt or os.path.join(args.load_dir, f"distilled_{args.load_iter}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    image_syn = ckpt["image"].to(args.device)
    text_syn = ckpt["text"].to(args.device)
    mask_syn = ckpt["mask"].to(args.device)

    agg_metrics = {"top1": [], "top5": [], "ece": []}
    for it_eval in range(args.num_eval):
        net_eval = CLIPModel_full(args)
        net_eval.image_encoder.load_state_dict(image_encoder_weights)
        net_eval.text_encoder.load_state_dict(text_encoder_weights)
        image_syn_eval = copy.deepcopy(image_syn.detach())
        text_syn_eval = copy.deepcopy(text_syn.detach())
        mask_syn_eval = copy.deepcopy(mask_syn.detach())

        _, metrics = evaluate_synset_zscls(
            it_eval=it_eval,
            net=net_eval,
            images_train=image_syn_eval,
            texts_train=text_syn_eval,
            mask_train=mask_syn_eval,
            cls_loader=testloader,
            cls_dataset=test_dataset,
            args=args,
        )
        agg_metrics["top1"].append(metrics["ZSCls/top1"])
        agg_metrics["top5"].append(metrics["ZSCls/top5"])
        agg_metrics["ece"].append(metrics["ZSCls/ece"])

    for key in ["top1", "top5", "ece"]:
        print(f"{key}: {np.mean(agg_metrics[key]):.2f} ({np.std(agg_metrics[key]):.2f})")

    results_table = print_results(
        agg_metrics,
        title=f"Zero-shot classification on {args.dataset} (ckpt={os.path.basename(ckpt_path)}, n_eval={args.num_eval})",
    )
    print(results_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate distilled checkpoint on zscls")

    parser.add_argument("--dataset", type=str, default="imagenet", help="zscls dataset")
    parser.add_argument("--num_queries", type=int, default=100, help="number of distilled samples (for default ckpt path)")
    parser.add_argument("--load_dir", type=str, default="results", help="run dir containing distilled_*.pt (used if --ckpt not set)")
    parser.add_argument("--load_iter", type=int, default=200, help="iteration of distilled set to load (used if --ckpt not set)")
    parser.add_argument("--ckpt", type=str, default=None, help="full path to checkpoint .pt (overrides load_dir/load_iter)")

    parser.add_argument("--image_encoder", type=str, default="nfnet")
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--image_pretrained", type=bool, default=True)
    parser.add_argument("--text_pretrained", type=bool, default=True)
    parser.add_argument("--image_trainable", type=bool, default=True)
    parser.add_argument("--text_trainable", type=bool, default=True)
    parser.add_argument("--proj_dim", type=int, default=2304)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size_train", type=int, default=128)
    parser.add_argument("--batch_size_test", type=int, default=128)

    parser.add_argument("--num_eval", type=int, default=1, help="number of evaluation runs to aggregate")
    parser.add_argument("--epoch_eval_train", type=int, default=100, help="epochs to train on synthetic data before eval")
    parser.add_argument("--lr_encoder_img", type=float, default=0.01)
    parser.add_argument("--lr_encoder_txt", type=float, default=0.01)
    parser.add_argument("--lr_proj_img", type=float, default=0.1)
    parser.add_argument("--lr_proj_txt", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
