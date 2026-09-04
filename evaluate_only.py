from collections import defaultdict
import os
import argparse

import numpy as np
import torch

import copy

from data import get_dataloaders
from src.epoch import evaluate_synset
from src.networks import CLIPModel_full



def main(args):
	''' organize the real train dataset '''  
	trainloader, testloader, train_dataset, test_dataset = get_dataloaders(args)

	print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
	print('Hyper-parameters: \n', args.__dict__)


	student_net = CLIPModel_full(args).to('cuda')
	student_net.eval()

	image_encoder_weights = copy.deepcopy(student_net.image_encoder.state_dict())
	text_encoder_weights = copy.deepcopy(student_net.text_encoder.state_dict())

	ckpt_path = f'{args.load_dir}/{args.dataset}/N{args.num_queries}/distilled_{args.load_iter}.pt'
	ckpt = torch.load(ckpt_path)
	image_syn = ckpt["image"].to(args.device)
	text_syn = ckpt["text"].to(args.device)
	mask_syn = ckpt["mask"].to(args.device)

	del student_net

	multi_eval_aggr_result = defaultdict(list)
	
	for it_eval in range(args.num_eval):
		net_eval = CLIPModel_full(args)

		net_eval.image_encoder.load_state_dict(image_encoder_weights)
		net_eval.text_encoder.load_state_dict(text_encoder_weights)

		image_syn_eval, text_syn_eval, mask_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(text_syn.detach()), copy.deepcopy(mask_syn.detach())

		_, _, best_val_result =  evaluate_synset(it_eval, net_eval, image_syn_eval, text_syn_eval, mask_syn_eval, testloader, test_dataset, args)

		for k, v in best_val_result.items():
			multi_eval_aggr_result[k].append(v)

	for key, values in multi_eval_aggr_result.items():
		print(f'{key}: {np.mean(values):.2f} ({np.std(values):.2f})')




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parameter Processing')


	parser.add_argument('--name', type=str, default='')
	parser.add_argument('--dataset', type=str, default='flickr', help='dataset')
	parser.add_argument('--num_queries', type=int, default=100, help='number of queries')
	parser.add_argument('--load_dir', type=str, default='results', help='directory to load synthetic set')
	parser.add_argument('--load_iter', type=int, default=200, help='distillation iteration of synthetic set to load')


	parser.add_argument('--image_encoder', type=str, default='nfnet',  help='image encoder')
	parser.add_argument('--text_encoder', type=str, default='bert', help='text encoder')
	parser.add_argument('--image_pretrained', type=bool, default=True, help='image_pretrained')
	parser.add_argument('--text_pretrained', type=bool, default=True, help='text_pretrained')
	parser.add_argument('--image_trainable', type=bool, default=True, help='image_trainable')
	parser.add_argument('--text_trainable', type=bool, default=True, help='text_trainable')
	parser.add_argument('--proj_dim', type=int, default=2304, help='projection dimension')


	parser.add_argument('--image_size', type=int, default=224, help='image_size')
	parser.add_argument('--ann_root', type=str, default='./data/Flickr30k_ann/', help='location of ann root')
	parser.add_argument('--image_root', type=str, default='distill_utils/data/Flickr30k/', help='location of image root')


	parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
	parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
	parser.add_argument('--batch_size_train', type=int, default=128, help='batch_size_train (for real)')
	parser.add_argument('--batch_size_test', type=int, default=128, help='batch_size_test')
	parser.add_argument('--lr_encoder_img', type=float, default=0.01, help='learning rate for updating network parameters')
	parser.add_argument('--lr_encoder_txt', type=float, default=0.01, help='learning rate for updating network parameters')
	parser.add_argument('--lr_proj_img', type=float, default=0.1, help='learning rate for updating network parameters')
	parser.add_argument('--lr_proj_txt', type=float, default=0.1, help='learning rate for updating network parameters')


	parser.add_argument('--wandb', action="store_true", help='wandb')
	parser.add_argument('--save', action="store_true", help='save')
	parser.add_argument('--device', type=str, default='cuda', help='device')
	parser.add_argument('--seed', type=int, default=0, help='seed')
	parser.add_argument('--logged_files', type=str, default='results', help='path to save synthetic dataset')

	
	args = parser.parse_args()

	BASE_DIR = '/path/to/datasets'
	if args.dataset == 'flickr':
		args.image_root = os.path.join(BASE_DIR, 'flickr30k')
	elif args.dataset == 'flickr8k':
		args.image_root = os.path.join(BASE_DIR, 'flickr8k')
	elif args.dataset == 'coco':
		args.image_root = os.path.join(BASE_DIR, 'coco2014')
	
	args.ann_root = os.path.join(BASE_DIR, 'annotations_retrieval')
	main(args)
