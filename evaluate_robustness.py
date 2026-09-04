import glob
import os
import copy
import math
import argparse
import random
import datetime
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.summary import logger
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import json 
from collections import defaultdict
import collections

import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt
from torch.utils.tensorboard import SummaryWriter


from data import get_dataset_flickr
from src.epoch import epoch_test, itm_eval, evaluate_synset_robustness
from src.networks import CLIPModel_full
from src.utils import DiffAugment, ParamDiffAug

import prettytable
from rich import print





	
EPS = 1e-8

@torch.no_grad()
def print_results(multi_eval_aggr_result, title='image-text retrieval results'):
	mean_result, std_result = defaultdict(), defaultdict()
	for k, v in multi_eval_aggr_result.items(): mean_result[k], std_result[k] = round(np.mean(v), 2), round(np.std(v), 2)


	results_table = prettytable.PrettyTable()
	results_table.float_format = ".2f"
	results_table.align = 'c'
	results_table.border = True
	results_table.field_names = ['Img R@1', 'Img R@5', 'Img R@10', 'Img R_Mean', 
								 'Txt R@1', 'Txt R@5', 'Txt R@10', 'Txt R_Mean', 
								 'R_Mean']
	results_table.title = title

	if prettytable.__version__ >= '3.16.0':
		results_table.add_divider()
	else:
		results_table.add_row(['-'*10]*9)         


	results_table.add_row(
		[f"{mean_result['img_r1']:.2f}", f"{mean_result['img_r5']:.2f}", f"{mean_result['img_r10']:.2f}", f"{mean_result['img_r_mean']:.2f}", 
		 f"{mean_result['txt_r1']:.2f}", f"{mean_result['txt_r5']:.2f}", f"{mean_result['txt_r10']:.2f}", f"{mean_result['txt_r_mean']:.2f}", 
		 f"{mean_result['r_mean']:.2f}"]
		)

	results_table.add_row(

		[f"± {std_result['img_r1']:.2f}", f"± {std_result['img_r5']:.2f}", f"± {std_result['img_r10']:.2f}", f"± {std_result['img_r_mean']:.2f}", 
		f"± {std_result['txt_r1']:.2f}", f"± {std_result['txt_r5']:.2f}", f"± {std_result['txt_r10']:.2f}", f"± {std_result['txt_r_mean']:.2f}", 
		f"± {std_result['r_mean']:.2f}"]
		)
	
	return results_table

def compute_ratio(angle_dict, k=2):
	ratio_dict = {}
	for key in angle_dict.keys():
		angle = np.deg2rad(angle_dict[key])
		ratio_dict[key] = k*np.cos(angle) / ((k-1)*np.cos(angle)+1+EPS)

	return ratio_dict 

def compute_angle(state_dict_1, state_dict_2, ref_state_dict, add_ignore_keys=[], return_cos=False, device='cuda'):
	ignore_keys = []
	return_dict = collections.OrderedDict()

	with torch.no_grad():
		for key in ref_state_dict:
			if key in ignore_keys:
				continue

			state_dict_1_val = state_dict_1[key]            
			state_dict_2_val = state_dict_2[key]                        
			ref_val = ref_state_dict[key]

			if not (state_dict_1_val.shape == state_dict_2_val.shape == ref_val.shape):
				continue 

			vector1 = (state_dict_1_val.to(device) - ref_val.to(device)).clone().detach()
			vector2 = (state_dict_2_val.to(device) - ref_val.to(device)).clone().detach()

			vector1 = vector1.float()
			vector2 = vector2.float()

			cosine_val = torch.sum(vector1 * vector2) / (sqrt(torch.sum(vector1 ** 2) * torch.sum(vector2 ** 2))+EPS)
			cosine_val = torch.clamp(cosine_val, min=-1., max=1.)
   
			cosine_val = cosine_val.to('cpu')
			if return_cos:
				return_dict[key] = cosine_val 
			else:
				return_dict[key] = np.rad2deg(torch.acos(cosine_val).detach().cpu())

	return return_dict

def merge(w1, w2, w0, ratio, device='cpu'):
	w12 = {}
	for key in w1.keys():                
		w12[key] = (w1[key].clone().to(device) + w2[key].clone().to(device)) / 2.

	w_merge = copy.deepcopy(w12)
	for key, r in ratio.items():        
		w_merge[key] = w12[key].clone().to(device) * r + w0[key].clone().to(device) * (1. - r)
	return w_merge

def _state_to_device(d, device='cuda', non_blocking=True, dtype=torch.float32):
	"""Move a (state_dict-like) mapping to device once, skipping non-floats/buffers."""
	out = {}
	for k, v in d.items():
		if torch.is_tensor(v) and v.is_floating_point():

			out[k] = v.to(device=device, dtype=dtype, non_blocking=non_blocking)
		else:

			out[k] = v
	return out

@torch.inference_mode()
def compute_cosine_dict(state1, state2, ref, device='cuda'):
	"""
	Return per-key cosine similarities between (state1-ref) and (state2-ref),
	computed entirely on GPU without acos/deg.
	"""
	cos_dict = collections.OrderedDict()



	s1 = state1
	s2 = state2
	r = ref

	for k in r.keys():
		if k not in s1 or k not in s2:
			continue
		v1, v2, vr = s1[k], s2[k], r[k]


		if (not torch.is_tensor(v1)) or (not torch.is_tensor(v2)) or (not torch.is_tensor(vr)):
			continue
		if (not v1.is_floating_point()) or (not v2.is_floating_point()) or (not vr.is_floating_point()):
			continue
		if v1.shape != v2.shape or v1.shape != vr.shape:
			continue

		a = (v1 - vr).clone().detach()
		b = (v2 - vr).clone().detach()


		dot = torch.dot(a.reshape(-1), b.reshape(-1))
		na  = torch.linalg.vector_norm(a)
		nb  = torch.linalg.vector_norm(b)
		cos = (dot / (na * nb + EPS)).clamp(-1.0, 1.0)
		cos_dict[k] = cos.detach().to('cpu')
	return cos_dict

def compute_ratio_from_cos(cos_dict, k=2.0):
	"""
	Same formula as your compute_ratio(angle_dict, k), but uses cos(theta) directly.
	ratio = k cosθ / ((k-1) cosθ + 1)
	"""
	out = {}
	for k_ in cos_dict.keys():
		c = float(cos_dict[k_])
		out[k_] = (k * c) / (((k - 1) * c) + 1.0 + EPS)
	return out

@torch.inference_mode()
def fast_merge(w1, w2, w0, ratio, device='cuda', dtype=torch.float32, non_blocking=True):
	"""
	빠른 버전 (당신의 merge와 동일한 수식):
	  1) w12 = (w1 + w2) / 2
	  2) w_merge[key] = r * w12 + (1 - r) * w0  (ratio에 key가 있으면), 없으면 w12 유지

	최적화:
	  - float 파라미터만 이동/연산
	  - 각 key당 .to() 1회
	  - r*w12 + (1-r)*w0 = w0 + r*(w12 - w0) 로 한 번에 연산
	"""
	w_merge = {}

	for k in w0.keys():

		if (k in w1) and (k in w2) \
		   and torch.is_tensor(w1[k]) and torch.is_tensor(w2[k]) and torch.is_tensor(w0[k]) \
		   and w1[k].is_floating_point() and w2[k].is_floating_point() and w0[k].is_floating_point() \
		   and (w1[k].shape == w2[k].shape == w0[k].shape):

			t1 = w1[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
			t2 = w2[k].to(device=device, dtype=dtype, non_blocking=non_blocking)
			t0 = w0[k].to(device=device, dtype=dtype, non_blocking=non_blocking)

			w12 = 0.5 * (t1 + t2)


			outk = w12


			if k in ratio:
				r = torch.as_tensor(ratio[k], dtype=dtype, device=device)

				outk = t0 + r * (w12 - t0)

			w_merge[k] = outk

		else:

			w_merge[k] = w0[k]

	return w_merge

def load_model_state_dict(state_dict, map_location='cpu'):
	state_dict = torch.load(state_dict, map_location=map_location)
	return state_dict

def make_distillation_model(args, img_expert_files, txt_expert_files, student_net, base_dir='./buffer/flickr8k/nfnet_bert/InfoNCE', file_format='replay_buffer', merge_image=True, merge_text=True, verbose=False):
	BASE_DIR = base_dir
	FILE_FORMAT = file_format	
	total_img_buffers = len(img_expert_files)-1

	FIX_EXPERT_VARY_EPOCH, VARY_EXPERT_VARY_EPOCH = False, False

 
	if FIX_EXPERT_VARY_EPOCH:
		EXPERT_NUM1 = random.randint(0, total_img_buffers)
		MAX_EPOCH = args.max_start_epoch
	
  
		EPOCH_NUM1 = random.choice(range(1, MAX_EPOCH+1))
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		img_expert_1 = load_model_state_dict(img_file, map_location='cuda')[0][EPOCH_NUM1]
		txt_expert_1 = load_model_state_dict(txt_file, map_location='cuda')[0][EPOCH_NUM1]
		
		epoch_pool = [i for i in range(1, MAX_EPOCH+1) if i != EPOCH_NUM1]
		EPOCH_NUM2 = random.choice(epoch_pool)
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		img_expert_2 = load_model_state_dict(img_file, map_location='cuda')[0][EPOCH_NUM2]
		txt_expert_2 = load_model_state_dict(txt_file, map_location='cuda')[0][EPOCH_NUM2]
  	
	elif VARY_EXPERT_VARY_EPOCH:
		EXPERT_NUM1 = random.randint(0, total_img_buffers)
		MAX_EPOCH = args.max_start_epoch

		EPOCH_NUM1 = random.choice(range(1, MAX_EPOCH+1)) 
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM1}.pt')
		img_expert_1 = load_model_state_dict(img_file, map_location='cuda')[0][EPOCH_NUM1]
		txt_expert_1 = load_model_state_dict(txt_file, map_location='cuda')[0][EPOCH_NUM1]
		
		EXPERT_NUM2 = random.randint(0, total_img_buffers)
		epoch_pool = [i for i in range(1, MAX_EPOCH+1) if i != EPOCH_NUM1]
		EPOCH_NUM2 = random.choice(epoch_pool)
		img_file = os.path.join(BASE_DIR, F'img_{FILE_FORMAT}_{EXPERT_NUM2}.pt')
		txt_file = os.path.join(BASE_DIR, F'txt_{FILE_FORMAT}_{EXPERT_NUM2}.pt')
		img_expert_2 = load_model_state_dict(img_file, map_location='cuda')[0][EPOCH_NUM2]
		txt_expert_2 = load_model_state_dict(txt_file, map_location='cuda')[0][EPOCH_NUM2]
		
	else:
		MIN_EPOCH = args.min_start_epoch
		MAX_EPOCH = args.max_start_epoch
  
		EXPERT_NUM1 = random.randint(0, 19)
		EPOCH_NUM1 = random.randint(MIN_EPOCH, MAX_EPOCH+1)
		img_file = os.path.join(BASE_DIR, f'img_replay_buffer_{EXPERT_NUM1}_{EPOCH_NUM1}.pth')
		txt_file = os.path.join(BASE_DIR, f'txt_replay_buffer_{EXPERT_NUM1}_{EPOCH_NUM1}.pth')
		img_expert_1 = load_model_state_dict(img_file, map_location='cuda')
		txt_expert_1 = load_model_state_dict(txt_file, map_location='cuda')
	
		EXPERT_NUM2 = random.randint(0, 19)
		if EPOCH_NUM1 == 0:
			EPOCH_NUM2 = random.randint(1, MAX_EPOCH+1)
		else:
			EPOCH_NUM2 = random.randint(MIN_EPOCH, MAX_EPOCH+1)
		img_file = os.path.join(BASE_DIR, f'img_replay_buffer_{EXPERT_NUM2}_{EPOCH_NUM2}.pth')
		txt_file = os.path.join(BASE_DIR, f'txt_replay_buffer_{EXPERT_NUM2}_{EPOCH_NUM2}.pth')
		img_expert_2 = load_model_state_dict(img_file, map_location='cuda')
		txt_expert_2 = load_model_state_dict(txt_file, map_location='cuda')
	
	method = 'FIX_EXPERT_VARY_EPOCH' if FIX_EXPERT_VARY_EPOCH else 'VARY_EXPERT_VARY_EPOCH' if VARY_EXPERT_VARY_EPOCH else 'RANDOM_EXPERT_RANDOM_EPOCH'

 

	assert isinstance(img_expert_1, dict) and isinstance(txt_expert_1, dict) and isinstance(img_expert_2, dict) and isinstance(txt_expert_2, dict)
	

	student_net.image_encoder.to('cuda')
	student_net.text_projection.to('cuda')
	img_initial = student_net.image_encoder.state_dict()
	txt_initial = student_net.text_projection.state_dict()
	
	if merge_image:



  
		cos_img = compute_cosine_dict(img_expert_1, img_expert_2, img_initial, device="cuda")
		ratio_img = compute_ratio_from_cos(cos_img, k=2.0)
		merged_img_model = fast_merge(img_expert_1, img_expert_2, img_initial, ratio_img, device="cuda")
	else:
		merged_img_model = None

	if merge_text:



  
		cos_txt = compute_cosine_dict(txt_expert_1, txt_expert_2, txt_initial, device="cuda")
		ratio_txt = compute_ratio_from_cos(cos_txt, k=2.0)
		merged_txt_model = fast_merge(txt_expert_1, txt_expert_2, txt_initial, ratio_txt, device="cuda")
	else:
		merged_txt_model = None



















	
	if verbose:
		return merged_img_model, merged_txt_model, (EXPERT_NUM1, EPOCH_NUM1), (EXPERT_NUM2, EPOCH_NUM2)
	else:
		return merged_img_model, merged_txt_model


def make_dir(p: str):
	if not os.path.exists(p):
		os.makedirs(p)

def to_jsonable(v):
	try:
		json.dumps(v)
		return v
	except TypeError:
		return {k: to_jsonable(x) for k, x in vars(v).items()} if hasattr(v, "__dict__") else str(v)

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
	return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	a = l2_normalize(a, dim=-1)
	b = l2_normalize(b, dim=-1)
	return a @ b.t()





def denormalize_clip(x):

	mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
	std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
	return x * std + mean

def to_grid_for_tb(x, nrow=8):

	x_denorm = denormalize_clip(x)
	x_denorm = torch.clamp(x_denorm, 0.0, 1.0)
	return make_grid(x_denorm.cpu(), nrow=nrow)

def _norm_t(it, T):
	if T is None or T <= 0:
		return 1.0
	t = it / float(T)
	return max(0.0, min(1.0, t))








def _load_clip_from_buffers(args):
	k = random.randint(0, args.num_buffers - 1)
	img_path = os.path.join(args.buffer_path, f'img_replay_buffer_{k}_10.pth')
	txt_path = os.path.join(args.buffer_path, f'txt_replay_buffer_{k}_10.pth')
	img_sd = torch.load(img_path, map_location='cuda')
	txt_sd = torch.load(txt_path, map_location='cuda')
 
	
	return img_sd, txt_sd, k





def clip_symmetric_nce_loss(img_feats: torch.Tensor, txt_feats: torch.Tensor, temperature: float = 0.07):
	logits = (img_feats @ txt_feats.t()) / temperature
	targets = torch.arange(img_feats.size(0), device=img_feats.device)
	loss_i2t = F.cross_entropy(logits, targets)
	loss_t2i = F.cross_entropy(logits.t(), targets)
	return 0.5 * (loss_i2t + loss_t2i)


def set_seed(seed):	
	import random
	from lightning.fabric import seed_everything
	seed_everything(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.empty_cache()
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	os.environ['PYTHONHASHSEED'] = str(seed)




def main(args):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	args.device = device
	set_seed(args.seed)


	trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)
	print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
	print('Hyper-parameters: \n', args.__dict__)


	make_dir(args.log_dir)
	current_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
	run_dir = os.path.join(args.log_dir, f"{current_time}_{args.name}")
	make_dir(run_dir)
	writer = SummaryWriter(log_dir=run_dir)
	writer.add_text("config/args", str(args), global_step=0)













 
 




	




 




 
 











	









	


	

	PATH = '/mnt/sdd/ddistillation/HGA_final_6798/final_distilled_data/RAHA_flickr8k_500_best40_rmean30.7__03_04_2026_112853/distilled_pairs_best_iter40.pt'

	
	print(PATH)
	eval_data = torch.load(PATH)







 
	image_syn = eval_data['image']
	text_syn  = eval_data['text']
	eval_target_it = eval_data['iter']
	if 'similarity_mat' in eval_data:
		similarity_mat = eval_data['similarity_mat']
	else:
		similarity_mat = None
	if 'mask' in eval_data:
		mask_syn = eval_data['mask']
	else:

		if text_syn.dim() == 3:
			mask_syn = torch.ones(text_syn.size(0), text_syn.size(1), dtype=torch.long, device=text_syn.device)
		else:
			mask_syn = torch.ones(text_syn.size(0), 1, dtype=torch.long, device=text_syn.device)


	student_net = CLIPModel_full(args).to(device)
	student_net.eval()
	image_encoder_weights = copy.deepcopy(student_net.image_encoder.state_dict())
	text_encoder_weights = copy.deepcopy(student_net.text_encoder.state_dict())
	del student_net

	image_syn = image_syn.to(device).detach()
	text_syn = text_syn.to(device).detach()
	mask_syn = mask_syn.to(device).detach()
		
		
	save_this_it = False


	print('Evaluation\nimage_model_train = %s, text_model_train = %s, iteration = %d'%(args.image_encoder, args.text_encoder, eval_target_it))
	
	print('DSA augmentation strategy: \n', args.dsa_strategy)
	print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
	
	multi_eval_aggr_result = defaultdict(list)
	

	for it_eval in range(args.num_eval):
		net_eval = CLIPModel_full(args)
		net_eval.image_encoder.load_state_dict(image_encoder_weights)
		net_eval.text_encoder.load_state_dict(text_encoder_weights)

		image_syn_eval = copy.deepcopy(image_syn.detach())
		text_syn_eval = copy.deepcopy(text_syn.detach())
		mask_syn_eval = copy.deepcopy(mask_syn.detach())

		_, _, best_val_result = evaluate_synset_robustness(
			it_eval=it_eval,
			net=net_eval,
			images_train=image_syn_eval,
			texts_train=text_syn_eval,
			mask_train=mask_syn_eval,
			testloader=testloader,
			test_dataset=test_dataset,
			args=args
		)
		
		for k, v in best_val_result.items():
			multi_eval_aggr_result[k].append(v)

	results_table = print_results(multi_eval_aggr_result, title=f'{args.name} results for {args.dataset}')
	print(f'Image encoder: {args.image_encoder}, Text encoder: {args.text_encoder}, Iteration: {eval_target_it}')
	logger.info(results_table)
	print(results_table)
	print(PATH)
	





	
	if save_this_it:
	
		with torch.no_grad():

			vis = image_syn.detach().cpu()
			img_log_dir = os.path.join(run_dir, 'images')
			make_dir(img_log_dir)
			save_path = os.path.join(img_log_dir, f"synthetic_images_{eval_target_it}.png")
			save_image(torch.clamp(denormalize_clip(vis)[:min(64, vis.size(0))], 0, 1), save_path, nrow=8)
			

			grid = to_grid_for_tb(vis[:min(64, vis.size(0))], nrow=8)
			writer.add_image("Synthetic/ImagesGrid", grid, eval_target_it)


			writer.add_histogram("Synthetic/Pixels", vis.flatten(), eval_target_it)
			writer.add_histogram("Synthetic/TextValues", text_syn.detach().cpu().flatten(), eval_target_it)

	
	writer.close()
	print("Done.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Joint Prototype Bank (JPB) Distillation for Image-Text Retrieval")


	parser.add_argument('--dataset', type=str, default='flickr', choices=['flickr', 'coco', 'flickr8k'])
	parser.add_argument('--data_path', type=str, default='./data/Flickr30k/')
	parser.add_argument('--image_root', type=str, default='./data/datasets/Flickr30k/')
	parser.add_argument('--ann_root', type=str, default='./data/Flickr30k_ann/')
	parser.add_argument('--log_dir', type=str, default='./log_cross_arch')


	parser.add_argument('--no_aug', action='store_true', help='no_aug')


	parser.add_argument('--feat_dim', type=int, default=768)


	parser.add_argument('--num_queries', type=int, default=100)


	parser.add_argument('--Iteration', type=int, default=3000)
	parser.add_argument('--batch_syn', type=int, default=64)
	parser.add_argument('--lr_img', type=float, default=100.0)
	parser.add_argument('--lr_txt', type=float, default=100.0)
	parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
	parser.add_argument('--momentum', type=float, default=0.5)
	parser.add_argument('--grad_clip', type=float, default=1.0)
	

	parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')


	parser.add_argument('--pix_init', type=str, default='real', choices=['real', 'noise'])
	parser.add_argument('--txt_init', type=str, default='real', choices=['real', 'noise'])
	parser.add_argument('--image_size', type=int, default=224)


	parser.add_argument('--image_encoder', type=str, default='nfnet', choices=['nfnet', 'nf_resnet50', 'nf_regnet', 'vit', 'dinov3'])
	parser.add_argument('--text_encoder',  type=str, default='bert', choices=['bert', 'clip', 'distilbert'])
	parser.add_argument('--proj_dim', type=int, default=2304, help='projection dimension (required by CLIPModel_full)')
	parser.add_argument('--text_pretrained',  type=bool, default=True)
	parser.add_argument('--image_pretrained', type=bool, default=True)
	parser.add_argument('--text_trainable',   type=bool, default=False)
	parser.add_argument('--image_trainable',  type=bool, default=True)
	parser.add_argument('--distill',          type=bool, default=True)
	parser.add_argument('--only_has_image_projection', type=bool, default=False, help='None')

	parser.add_argument('--temperature', type=float, default=0.07)


	parser.add_argument('--eval_it', type=int, default=100, help='evaluation interval')
	parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train on synthetic for eval')
	parser.add_argument('--eval_eval_it', type=int, default=10, help='evaluation frequency during eval training')
	parser.add_argument('--num_eval', type=int, default=5, help='repeat eval training')
	parser.add_argument('--batch_size_train', type=int, default=64, help='batch_size_train')
	parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
	parser.add_argument('--batch_size_test', type=int, default=64, help='batch_size_test')
	parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='eval-time image LR')
	parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='eval-time text LR')
	parser.add_argument('--lr_encoder_img', type=float, default=0.01, help='learning rate for image encoder (evaluate_synset)')
	parser.add_argument('--lr_encoder_txt', type=float, default=0.01, help='learning rate for text encoder (evaluate_synset)')
	parser.add_argument('--lr_proj_img', type=float, default=0.1, help='learning rate for image projection (evaluate_synset)')
	parser.add_argument('--lr_proj_txt', type=float, default=0.1, help='learning rate for text projection (evaluate_synset)')
	parser.add_argument('--loss_type', type=str, default="InfoNCE", help='InfoNCE or WBCE')

	

	parser.add_argument('--buffer_path', type=str, default='/mnt/hoyong3/dataset-distil/LoRS_Distill/buffer_my_seed0123/flickr30k/nfnet_bert/InfoNCE')
	parser.add_argument('--num_buffers', type=int, default=20)
	parser.add_argument('--teacher_resample', type=int, default=50)


	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--name', type=str, default='cross_arch')
	parser.add_argument('--log_it', type=int, default=10)
	parser.add_argument('--save_it', type=int, default=200)
	

	parser.add_argument('--syn_init', type=str, default='kmeans', choices=['random', 'kmeans'])
	

	parser.add_argument('--cluster_by', type=str, default='image_text', choices=['image', 'text', 'image_text'])
	parser.add_argument('--cluster_mode', type=str, default='cosine', choices=['cosine','euclidean'])


	parser.add_argument('--w_nce', type=float, default=0.1, help='weight for image-text NCE loss')
 

	parser.add_argument('--w_sph_mmd', type=float, default=1.0,
						help='Weight for spherical MMD loss.')
	parser.add_argument('--sph_mmd_sigma', type=float, default=0.5,
						help='Sigma of geodesic RBF kernel for spherical MMD.')



	parser.add_argument('--w_cgap_y', type=float, default=0.0, help='Weight for conditional GAP matching g|y.')
	parser.add_argument('--w_cgap_x', type=float, default=0.0, help='Weight for conditional GAP matching g|x.')
	parser.add_argument('--c_sigma_y', type=float, default=0.5, help='Kernel sigma for conditioning on Y.')
	parser.add_argument('--c_sigma_x', type=float, default=0.5, help='Kernel sigma for conditioning on X.')
	parser.add_argument('--cgap_it_max', type=int, default=5000, help='Maximum iteration that cgap weight increase.')
	parser.add_argument('--c_sigma_g', type=float, default=0.5, help='Kernel sigma for GAP vector G.')

	parser.add_argument('--w_cgap_reg', type=float, default=0.0, help='Weight for conditional regularizer loss.')
	parser.add_argument('--w_cgap_rep', type=float, default=0.0, help='Weight for conditional representation loss.')
	
	parser.add_argument('--cgap_schedule', type=str, default='constant', choices=['constant', 'linear', 'log', 'exp', 'sigmoid'],
						help='cgap weight schedule that arise from 0 to w_cgap for cgap_it_max iterations')
	parser.add_argument('--cgap_log_k', type=float, default=9.0, help='log(1+k*a)')
	parser.add_argument('--cgap_exp_k', type=float, default=5.0, help='e^(kt-1) / (e^k - 1)')
	parser.add_argument('--cgap_sig_k', type=float, default=10.0, help='1/(1+e^(-k(t-0.5)))')
	parser.add_argument('--text_embed_dir', type=str, default='text_embeds', help='text embed npz file directory')
	parser.add_argument('--min_start_epoch', type=int, default=1, help='max epoch we can start at')
	parser.add_argument('--max_start_epoch', type=int, default=3, help='max epoch we can start at')
 
 
	parser.add_argument('--init_model_method', type=str, default='default', choices=['default', 'mixed', 'naive'])
 
	parser.add_argument('--eval_target_path', type=str, default='/mnt/hoyong3/dataset-distil/LoRS_Distill/log_cross_arch/11-04_03-14-54_flickr30k_evModelRand_Mws_nce1_wMMD5', help='path to load distilled data for evaluation')
	parser.add_argument('--target_it', type=int, default=3000, help='which iteration distilled data to load for evaluation')
	
	
	

	parser.add_argument('--image_perturb', type=bool, default=False, help='whether to perturb image')
	parser.add_argument('--text_embed_perturb', type=bool, default=False, help='whether to perturb text embed')
	
	args = parser.parse_args()
	


	args.image_perturb = False
	args.text_embed_perturb = True
	






	args.dsa_param = ParamDiffAug()
	args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

	BASE_DIR = '/mnt/sdc/mdd_datasets'
	if args.dataset == 'flickr':
		args.image_root = os.path.join(BASE_DIR, 'flickr30k')
	elif args.dataset == 'flickr8k':
		args.image_root = os.path.join(BASE_DIR, 'flickr8k')
	elif args.dataset == 'coco':
		args.image_root = os.path.join(BASE_DIR, 'coco2014')
	
	args.ann_root = os.path.join(BASE_DIR, 'annotations_retrieval')

	if args.buffer_path is None:
		args.buffer_path = f'./buffer/{args.dataset}/{args.image_encoder}_{args.text_encoder}/{args.loss_type}'
	
	args.log_dir = os.path.join(args.log_dir, __file__.split('/')[-1].split('.')[0])

	torch.set_float32_matmul_precision('high')
	main(args)











