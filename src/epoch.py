'''
 * part of the code (i.e. def epoch_test() and itm_eval()) is from: https://github.com/salesforce/BLIP/blob/main/train_retrieval.py#L69
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
from math import ceil
import time
import datetime
import copy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from rich import print
import wandb

from data import get_dataloaders
from src.networks import CLIPModel_full
from src.utils import *
from src.vl_distill_utils import get_images_texts




def epoch(e, dataloader, net, optimizer, args):
	net = net.to(args.device)
	net.train()
	loss_avg, acc_avg, num_exp = 0, 0, 0

	for i, data in (enumerate(dataloader)):
		image, text, mask = data

		image = image.to(args.device)
		n_b = image.shape[0]

		loss, acc = net(image, text, mask)

		loss_avg += loss.item() * n_b
		acc_avg += acc
		num_exp += n_b

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	loss_avg /= num_exp
	acc_avg /= num_exp

	return loss_avg, acc_avg



@torch.no_grad()
def epoch_test(dataloader, testset, model, device, args):
	model.eval() 
	logit_scale = model.logit_scale.detach()
	start_time = time.time()

	batch_size = 1000
	text_embeds_list = []
	for i in range(0, len(testset.text), batch_size):
		batch_texts = testset.text[i:i+batch_size]
		output = model.text_encoder.forward_raw(batch_texts)
		
		output = output.float()
		text_feat = model.text_projection(output)
		text_embeds_batch = text_feat / text_feat.norm(dim=1, keepdim=True)
		text_embeds_list.append(text_embeds_batch)
	text_embeds = torch.cat(text_embeds_list, dim=0)

	image_embeds = []
	for image, img_id in dataloader: 
		image_feat = model.image_encoder(image.to(device))
		im_embed = model.image_projection(image_feat.float())
		im_embed = im_embed / im_embed.norm(dim=1, keepdim=True)
		image_embeds.append(im_embed)
	image_embeds = torch.cat(image_embeds,dim=0)    
		
	sims_matrix = logit_scale.exp() * image_embeds @ text_embeds.t()
	score_matrix_i2t = torch.full((len(image_embeds),len(text_embeds)),-100.0).to(device) #torch.Size([1000, 5000])
	for i, sims in enumerate(sims_matrix[0:sims_matrix.size(0) + 1]): 
		topk_sim, topk_idx = sims.topk(k=128, dim=0)
		score_matrix_i2t[i,topk_idx] = topk_sim #i:0-999, topk_idx:0-4999, find top k (k=128) similar text for each image
	
	sims_matrix = sims_matrix.t()
	score_matrix_t2i = torch.full((len(text_embeds),len(image_embeds)),-100.0).to(device)
	for i,sims in enumerate(sims_matrix[0:sims_matrix.size(0) + 1]): 
		topk_sim, topk_idx = sims.topk(k=128, dim=0)
		score_matrix_t2i[i,topk_idx] = topk_sim

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	# print('\nEvaluation time {}'.format(total_time_str)) 

	return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def epoch_test_robustness(
	dataloader,
	testset,
	model,
	device,
	args,
	num_eval=5
):
	"""
	Like epoch_test but optionally adds perturbations to images and/or text embeddings
	for robustness evaluation. Perturbation types: PGD, awgn, quantization (text);
	PGD, awgn, jpeg (image).
	"""
	model.eval()
	logit_scale = model.logit_scale.detach()
	start_time = time.time()

	################################################################

	################################################################
	
	TEXT_PERTURBATION_TYPE = None
	IMAGE_PERTURBATION_TYPE = None

	if args.image_perturb:
		# IMAGE_PERTURBATION_TYPE = 'pgd'
		# IMAGE_PERTURBATION_TYPE = 'jpeg'
		IMAGE_PERTURBATION_TYPE = 'awgn'

	if args.text_embed_perturb:
		TEXT_PERTURBATION_TYPE = 'pgd'
		# TEXT_PERTURBATION_TYPE = 'awgn'
		# TEXT_PERTURBATION_TYPE = 'quantization'
		
	print(f"\nTEXT_PERTURBATION_TYPE: [yellow]{TEXT_PERTURBATION_TYPE}[/yellow]")
	print(f"IMAGE_PERTURBATION_TYPE: [yellow]{IMAGE_PERTURBATION_TYPE}[/yellow]")
	
	batch_size = 1000
	text_embeds_list = []
	for i in range(0, len(testset.text), batch_size):
		batch_texts = testset.text[i : i + batch_size]
		output = model.text_encoder.forward_raw(batch_texts)
		output = output.float()
		text_feat = model.text_projection(output)
		txt_embed = text_feat
		text_embeds_batch = txt_embed / txt_embed.norm(dim=1, keepdim=True)
		text_embeds_list.append(text_embeds_batch)
	text_embeds = torch.cat(text_embeds_list, dim=0)

	if args.text_embed_perturb:
		if TEXT_PERTURBATION_TYPE == "pgd":
			pgd_epsilon_txt = 0.05
			pgd_alpha_txt = pgd_epsilon_txt / 4
			pgd_iters_txt = 10
			txt_embed = text_embeds.clone().detach()
			txt_embed_orig = txt_embed.clone().detach()
			txt_embed_orig_norm = txt_embed_orig / txt_embed_orig.norm(dim=1, keepdim=True)
			delta_txt = torch.empty_like(txt_embed).uniform_(-pgd_epsilon_txt, pgd_epsilon_txt)
			delta_txt.requires_grad_(True)
			for _ in range(pgd_iters_txt):
				txt_embed_pert = txt_embed_orig + delta_txt
				with torch.enable_grad():
					txt_embed_pert.requires_grad_(True)
					txt_embed_pert_norm = txt_embed_pert / txt_embed_pert.norm(dim=1, keepdim=True)
					loss_pgd_txt = -(txt_embed_pert_norm * txt_embed_orig_norm).sum(dim=1).mean()
					grad_txt = torch.autograd.grad(loss_pgd_txt, txt_embed_pert, create_graph=False)[0]
				with torch.no_grad():
					delta_txt.data = delta_txt.data + pgd_alpha_txt * grad_txt.sign()
					delta_txt.data = torch.clamp(delta_txt.data, -pgd_epsilon_txt, pgd_epsilon_txt)
			text_embeds = (txt_embed_orig + delta_txt.detach()) / (
				(txt_embed_orig + delta_txt.detach()).norm(dim=1, keepdim=True)
			)
		elif TEXT_PERTURBATION_TYPE == "awgn":
			from src.robustness_utils import add_awgn_text_embed
			text_embeds = add_awgn_text_embed(text_embeds, sigma=0.01)
		elif TEXT_PERTURBATION_TYPE == "quantization":
			from src.robustness_utils import quantize_unit_embed
			text_embeds = quantize_unit_embed(text_embeds, bits=4)
		else:
			raise ValueError(f"Invalid text perturbation type: {TEXT_PERTURBATION_TYPE}")

	text_embeds = text_embeds.to(device)

	image_embeds = []
	for image, img_id in dataloader:
		if args.image_perturb:
			image = image.to(device)
			if IMAGE_PERTURBATION_TYPE == "pgd":
				pgd_epsilon = 2.0 / 255
				pgd_alpha = pgd_epsilon / 4
				pgd_iters = 10
				image = image.to(device).clone().detach()
				image_orig = image.clone().detach()
				with torch.no_grad():
					image_feat_orig = model.image_encoder(image_orig)
					im_embed_orig = model.image_projection(image_feat_orig.float())
					im_embed_orig = im_embed_orig / im_embed_orig.norm(dim=1, keepdim=True)
				delta = torch.empty_like(image).uniform_(-pgd_epsilon, pgd_epsilon)
				delta = torch.clamp(image_orig + delta, 0, 1) - image_orig
				delta.requires_grad_(True)
				for _ in range(pgd_iters):
					image_pert = image_orig + delta
					image_pert = torch.clamp(image_pert, 0, 1)
					with torch.enable_grad():
						image_pert.requires_grad_(True)
						image_feat = model.image_encoder(image_pert)
						im_embed = model.image_projection(image_feat.float())
						im_embed = im_embed / im_embed.norm(dim=1, keepdim=True)
						loss_pgd = -(im_embed * im_embed_orig).sum(dim=1).mean()
						grad = torch.autograd.grad(loss_pgd, image_pert, create_graph=False)[0]
					with torch.no_grad():
						delta.data = delta.data + pgd_alpha * grad.sign()
						delta.data = torch.clamp(delta.data, -pgd_epsilon, pgd_epsilon)
						delta.data = torch.clamp(image_orig + delta.data, 0, 1) - image_orig
				image = torch.clamp(image_orig + delta.detach(), 0, 1)
			elif IMAGE_PERTURBATION_TYPE == "awgn":
				from src.robustness_utils import add_gaussian_noise_image
				image = add_gaussian_noise_image(image, sigma=0.01)
			elif IMAGE_PERTURBATION_TYPE == "jpeg":
				from src.robustness_utils import jpeg_compress_batch
				image = jpeg_compress_batch(image, quality=75)
			else:
				raise ValueError(f"Invalid image perturbation type: {IMAGE_PERTURBATION_TYPE}")

		image_feat = model.image_encoder(image.to(device))
		im_embed = model.image_projection(image_feat.float())
		im_embed = im_embed / im_embed.norm(dim=1, keepdim=True)
		image_embeds.append(im_embed)
	image_embeds = torch.cat(image_embeds, dim=0)

	sims_matrix = logit_scale.exp() * image_embeds @ text_embeds.t()
	score_matrix_i2t = torch.full(
		(len(image_embeds), len(text_embeds)), -100.0
	).to(device)
	for i, sims in enumerate(sims_matrix[0 : sims_matrix.size(0) + 1]):
		topk = 128 if sims.shape[0] > 128 else sims.shape[0]
		topk_sim, topk_idx = sims.topk(k=topk, dim=0)
		score_matrix_i2t[i, topk_idx] = topk_sim
	sims_matrix = sims_matrix.t()
	score_matrix_t2i = torch.full(
		(len(text_embeds), len(image_embeds)), -100.0
	).to(device)
	for i, sims in enumerate(sims_matrix[0 : sims_matrix.size(0) + 1]):
		topk = 128 if sims.shape[0] > 128 else sims.shape[0]
		topk_sim, topk_idx = sims.topk(k=topk, dim=0)
		score_matrix_t2i[i, topk_idx] = topk_sim

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
	
	#Images->Text 
	ranks = np.zeros(scores_i2t.shape[0])
	# print("TR: ", len(ranks))
	for index, score in enumerate(scores_i2t):
		inds = np.argsort(score)[::-1]
		# Score
		rank = 1e20
		for i in img2txt[index]:
			tmp = np.where(inds == i)[0][0]
			if tmp < rank:
				rank = tmp
		ranks[index] = rank

	# Compute metrics
	tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
	#Text->Images 
	ranks = np.zeros(scores_t2i.shape[0])
	# print("IR: ", len(ranks))
	
	for index,score in enumerate(scores_t2i):
		inds = np.argsort(score)[::-1]
		ranks[index] = np.where(inds == txt2img[index])[0][0]

	# Compute metrics
	ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

	tr_mean = (tr1 + tr5 + tr10) / 3
	ir_mean = (ir1 + ir5 + ir10) / 3
	r_mean = (tr_mean + ir_mean) / 2

	eval_result =  {'txt_r1': tr1,
					'txt_r5': tr5,
					'txt_r10': tr10,
					'txt_r_mean': tr_mean,
					'img_r1': ir1,
					'img_r5': ir5,
					'img_r10': ir10,
					'img_r_mean': ir_mean,
					'r_mean': r_mean}
	return eval_result


def evaluate_synset(it_eval, net, images_train, texts_train, mask_train, testloader, test_dataset, args, return_loss=False, num_eval=5):
	
	net = net.to(args.device)
	images_train = images_train.to(args.device)
	texts_train = texts_train.to(args.device)
	mask_train = mask_train.to(args.device)
	Epoch = int(args.epoch_eval_train)

	optimizer = torch.optim.SGD([
		{'params': net.image_encoder.parameters(), 'lr': args.lr_encoder_img},
		{'params': net.image_projection.parameters(), 'lr': args.lr_proj_img},
		{'params': net.text_encoder.parameters(), 'lr': args.lr_encoder_txt},
		{'params': net.text_projection.parameters(), 'lr': args.lr_proj_txt},
	], lr=0, momentum=0.9, weight_decay=0.0005)
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[Epoch//2+1], gamma=0.1)

	dst_train = TensorDataset(images_train, texts_train, mask_train)
	train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size_train, shuffle=True, num_workers=0)

	start = time.time()
	acc_train_list = []
	loss_train_list = []

	# eval_epochs = [Epoch]
	# eval_freq = Epoch // 10
	# eval_epochs = list(range(0, Epoch+1, eval_freq)) + [Epoch] # eval 10 times in total

	best_val_result = None
	for ep in tqdm(range(1, Epoch+1), ncols=60):
		loss_train, acc_train = epoch(ep, train_loader, net, optimizer, args)
		acc_train_list.append(acc_train)
		loss_train_list.append(loss_train)
		if ep > 0 and ep % args.eval_eval_it == 0: #in eval_epochs:
			with torch.no_grad():
				score_val_i2t, score_val_t2i = epoch_test(testloader, test_dataset, net, args.device, args)
				val_result = itm_eval(score_val_i2t, score_val_t2i, testloader.dataset.txt2img, testloader.dataset.img2txt) 
				
				
				print("[Eval_{it:02d}] Ep{ep} | Image R@1={img_r1:.2f} R@5={img_r5:.2f} R@10={img_r10:.2f} | Text R@1={txt_r1:.2f} R@5={txt_r5:.2f} R@10={txt_r10:.2f} | Mean={r_mean:.2f}".format(
					it=it_eval, Epoch=Epoch, ep=ep, **val_result
				))
				if best_val_result is None or val_result["r_mean"] > best_val_result["r_mean"]:
					best_val_result = val_result
					
		lr_scheduler.step()

	time_train = time.time() - start
	
	return net, acc_train_list, best_val_result


def evaluate_synset_robustness(
	it_eval,
	net,
	images_train,
	texts_train,
	mask_train,
	testloader,
	test_dataset,
	args
):
	"""
	Same as evaluate_synset but runs retrieval evaluation under perturbations
	via epoch_test_robustness (image and/or text embedding perturbation).
	"""
	net = net.to(args.device)
	images_train = images_train.to(args.device)
	texts_train = texts_train.to(args.device)
	mask_train = mask_train.to(args.device)
	Epoch = int(args.epoch_eval_train)

	optimizer = torch.optim.SGD([
		{'params': net.image_encoder.parameters(), 'lr': args.lr_encoder_img},
		{'params': net.image_projection.parameters(), 'lr': args.lr_proj_img},
		{'params': net.text_encoder.parameters(), 'lr': args.lr_encoder_txt},
		{'params': net.text_projection.parameters(), 'lr': args.lr_proj_txt},
	], lr=0, momentum=0.9, weight_decay=0.0005)
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
		optimizer, milestones=[Epoch // 2 + 1], gamma=0.1
	)

	dst_train = TensorDataset(images_train, texts_train, mask_train)
	train_loader = torch.utils.data.DataLoader(
		dst_train,
		batch_size=args.batch_size_train,
		shuffle=True,
		num_workers=0,
	)

	start = time.time()
	acc_train_list = []
	loss_train_list = []
	best_val_result = None

	for ep in tqdm(range(1, Epoch + 1), ncols=60):
		loss_train, acc_train = epoch(ep, train_loader, net, optimizer, args)
		acc_train_list.append(acc_train)
		loss_train_list.append(loss_train)
		if ep > 0 and ep % args.eval_eval_it == 0:

			with torch.no_grad():
				score_val_i2t, score_val_t2i = epoch_test_robustness(
					testloader,
					test_dataset,
					net,
					args.device,
					args
				)
				val_result = itm_eval(
					score_val_i2t,
					score_val_t2i,
					testloader.dataset.txt2img,
					testloader.dataset.img2txt,
				)
				print(
					"[Eval_{it:02d}/{Epoch}] Ep{ep} (robustness) | "
					"Image R@1={img_r1:.2f} R@5={img_r5:.2f} R@10={img_r10:.2f} | "
					"Text R@1={txt_r1:.2f} R@5={txt_r5:.2f} R@10={txt_r10:.2f} | Mean={r_mean:.2f}".format(
						it=it_eval, Epoch=Epoch, ep=ep, **val_result
					)
				)
				if best_val_result is None or val_result["r_mean"] > best_val_result["r_mean"]:
					best_val_result = val_result
		lr_scheduler.step()

	time_train = time.time() - start
	return net, acc_train_list, best_val_result


def _compute_topk_and_ece(logits: torch.Tensor, labels: torch.Tensor, num_bins: int = 15):
	"""
	Compute top-1, top-5 accuracy and Expected Calibration Error (ECE).
	logits: [N, C], labels: [N]
	"""
	with torch.no_grad():
		probs = torch.softmax(logits, dim=1)
		confs, preds = probs.max(dim=1)
		correct = preds.eq(labels)

		# Top-1
		top1 = correct.float().mean().item() * 100.0

		# Top-5
		_top5_vals, top5_idx = probs.topk(5, dim=1)
		top5_correct = top5_idx.eq(labels.unsqueeze(1)).any(dim=1)
		top5 = top5_correct.float().mean().item() * 100.0

		# ECE
		bins = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=probs.device)
		ece = torch.zeros((), device=probs.device)
		for i in range(num_bins):
			bin_lower = bins[i]
			bin_upper = bins[i + 1]
			in_bin = (confs > bin_lower) & (confs <= bin_upper)
			if in_bin.any():
				prop_in_bin = in_bin.float().mean()
				acc_in_bin = correct[in_bin].float().mean()
				conf_in_bin = confs[in_bin].mean()
				ece += torch.abs(acc_in_bin - conf_in_bin) * prop_in_bin
		ece = ece.item()

	return top1, top5, ece


def evaluate_synset_zscls(
	it_eval,
	net,
	images_train,
	texts_train,
	mask_train,
	cls_loader,
	cls_dataset,
	args,
	num_bins: int = 15,
):
	"""
	Train the net on distilled synthetic data (same epochs as evaluate_synset)
	and evaluate classification on cls_loader / cls_dataset.

	Metrics: top-1, top-5, and ECE. Metrics are also logged to wandb.
	"""
	device = args.device
	net = net.to(device)
	images_train = images_train.to(device)
	texts_train = texts_train.to(device)
	mask_train = mask_train.to(device)
	Epoch = int(args.epoch_eval_train)

	optimizer = torch.optim.SGD(
		[
			{"params": net.image_encoder.parameters(), "lr": args.lr_encoder_img},
			{"params": net.image_projection.parameters(), "lr": args.lr_proj_img},
			{"params": net.text_encoder.parameters(), "lr": args.lr_encoder_txt},
			{"params": net.text_projection.parameters(), "lr": args.lr_proj_txt},
		],
		lr=0,
		momentum=0.9,
		weight_decay=0.0005,
	)
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
		optimizer, milestones=[Epoch // 2 + 1], gamma=0.1
	)

	dst_train = TensorDataset(images_train, texts_train, mask_train)
	train_loader = torch.utils.data.DataLoader(
		dst_train,
		batch_size=args.batch_size_train,
		shuffle=True,
		num_workers=0,
	)

	print(
		f"[zscls train] Eval #{it_eval + 1} | epochs={Epoch} | "
		f"{len(dst_train)} synthetic samples",
		flush=True,
	)
	for ep in tqdm(range(1, Epoch + 1), ncols=60):
		loss_train, acc_train = epoch(ep, train_loader, net, optimizer, args)
		lr_scheduler.step()

	# Build zero-shot classifier weights from class names using the same prompt template
	# as in PromptedClassificationDataset.
	classes = getattr(cls_dataset, "classes", None)
	prompt_template = getattr(cls_dataset, "prompt_template", "a photo of a {}.")
	if classes is None:
		raise ValueError("cls_dataset must expose a 'classes' attribute for zscls.")

	texts = [prompt_template.format(c) for c in classes]
	net.eval()
	with torch.no_grad():
		# Encode class texts
		text_emb = net.text_encoder.forward_raw(texts, device=device)
		text_emb = text_emb.float()
		text_feat = net.text_projection(text_emb)
		text_feat = text_feat / (text_feat.norm(dim=1, keepdim=True) + 1e-12)

		all_logits = []
		all_labels = []
		for batch in tqdm(cls_loader, desc="zscls-eval", ncols=60):
			# PromptedClassificationDataset returns (image, caption, label)
			images, _captions, labels = batch
			images = images.to(device)
			labels = labels.to(device).long()

			img_feat = net.image_encoder(images)
			img_feat = net.image_projection(img_feat.float())
			img_feat = img_feat / (img_feat.norm(dim=1, keepdim=True) + 1e-12)

			logit_scale = net.logit_scale.exp()
			logits = logit_scale * img_feat @ text_feat.t()

			all_logits.append(logits.detach())
			all_labels.append(labels.detach())

		all_logits = torch.cat(all_logits, dim=0)
		all_labels = torch.cat(all_labels, dim=0)

	top1, top5, ece = _compute_topk_and_ece(all_logits, all_labels, num_bins=num_bins)

	metrics = {
		"ZSCls/top1": top1,
		"ZSCls/top5": top5,
		"ZSCls/ece": ece,
	}
	# Use it_eval as the logging step for zscls eval
	try:
		wandb.log(metrics, step=int(it_eval))
	except Exception as e:
		print(f"[warn] wandb.log failed in evaluate_synset_zscls: {e}")

	print(
		f"[zscls eval] top1={top1:.2f}%, top5={top5:.2f}%, ECE={ece:.4f}",
		flush=True,
	)

	return net, metrics

