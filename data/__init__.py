import json
import os
import re
from math import ceil
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets as tv_datasets
from torchvision.transforms.functional import InterpolationMode

from data.randaugment import RandomAugment
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.coco_dataset import coco_train, coco_retrieval_eval
from data.flickr8k_dataset import flickr8k_train, flickr8k_retrieval_eval
from data.cc3m_595k_dataset import cc3m_595k_llava_train, cc3m_595k_llava_retrieval_eval
from data.coyo_dataset import COYO700MSubset, COYO700MSubsetConfig, COYO700MSubsetRetrievalEval
from data.mmcelebahq_dataset import mmcelebahq_train, mmcelebahq_retrieval_eval


# from src.vl_distill_utils import coreset

__all__ = [
	"create_loader",
	"get_dataloaders",
	"get_dataset_flickr",
	"create_zscls_dataset",
	"get_zscls_dataloaders",
	"get_zscls_paths",
]

ZSCLS_DATASETS = {"imagenet", "imagenet-r", "imagenet-a", "cub", "stcar", "air", "cifar100"}
ZSCLS_BASE_DIR = os.environ.get("ZSCLS_BASE_DIR", "/path/to/datasets")


def get_zscls_paths(dataset_name: str, base_dir: Optional[str] = None) -> dict:
	"""
	Return paths for a zscls dataset from data/{dataset_name}.py.
	BIA-style: image_root, ann_root (train.txt/test.txt), train_root, val_root.
	base_dir: e.g. DATA_ROOT_DIR; if None, uses ZSCLS_BASE_DIR.
	"""
	base = base_dir or ZSCLS_BASE_DIR
	mod_name = dataset_name.replace("-", "_")
	try:
		mod = __import__(f"data.{mod_name}", fromlist=["get_paths"])
		get_paths = getattr(mod, "get_paths", None)
		if get_paths is not None:
			return get_paths(base)
		# Fallback: use IMAGE_ROOT from module
		image_root = getattr(mod, "IMAGE_ROOT", None)
		if image_root:
			return {"image_root": image_root, "ann_root": None}
	except ImportError:
		pass
	return {"image_root": os.path.join(base, dataset_name), "ann_root": None}


def _load_imagenet_class_index() -> Dict[str, str]:
	"""Load ImageNet synset_id -> class_name mapping from keras-vis imagenet_class_index.json."""
	path = os.path.join(os.path.dirname(__file__), "imagenet_class_index.json")
	try:
		with open(path, "r") as f:
			idx = json.load(f)
		# idx: {"0": ["n01440764", "tench"], ...} -> synset_id -> class_name
		return {v[0]: v[1] for v in idx.values()}
	except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
		raise FileNotFoundError(
			f"Could not load ImageNet class index from {path}. "
			"Download from https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json"
		) from e


def _clean_zscls_class_name(ds_name: str, raw_name: str) -> str:
	"""Remove leading numerical prefix from class names for CUB, STCAR, Aircraft."""
	if ds_name not in ("cub", "stcar", "air"):
		return raw_name
	# Remove leading "001." or "001 " pattern (digits + optional dot + optional space),
	# and replace underscores with spaces
	cleaned_name = re.sub(r"^\d+\.?\s*", "", raw_name)
	cleaned_name = cleaned_name.replace('_', ' ')
	return cleaned_name.strip() or raw_name


# def create_sampler(datasets, shuffles, num_tasks, global_rank):
#     samplers = []
#     for dataset,shuffle in zip(datasets,shuffles):
#         sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
#         samplers.append(sampler)
#     return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns, pin_memory=True):
	loaders = []
	for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
		if is_train:
			shuffle = (sampler is None)
			drop_last = True
		else:
			shuffle = False
			drop_last = False
		loader = DataLoader(
			dataset,
			batch_size=bs,
			num_workers=n_worker,
			pin_memory=pin_memory,
			sampler=sampler,
			shuffle=shuffle,
			collate_fn=collate_fn,
			drop_last=drop_last,
		)              
		loaders.append(loader)
	return loaders    

def get_dataloaders(args):
	min_scale = 0.5
	normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
	transform_train = transforms.Compose([                        
			transforms.RandomResizedCrop(args.image_size,scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
			transforms.RandomHorizontalFlip(),
			RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
											  'ShearX', 'ShearY', 'TranslateX', 'TranslateY']),     
			transforms.ToTensor(),
			normalize,
		])     
	transform_test = transforms.Compose([
		transforms.Resize((args.image_size, args.image_size),interpolation=InterpolationMode.BICUBIC),
		transforms.ToTensor(),
		normalize,
		])
 
	# transform_train = transform_test # for visualization
 
	if args.dataset == 'flickr':          
		train_dataset = flickr30k_train(transform_train, args.image_root, args.ann_root)
		val_dataset = flickr30k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val') 
		test_dataset = flickr30k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')         
			
	elif args.dataset == 'coco':             
		train_dataset = coco_train(transform_train, args.image_root, args.ann_root)
		val_dataset = coco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val') 
		test_dataset = coco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')         
		
	elif args.dataset == 'cc3m_595k_llava':
		train_dataset = cc3m_595k_llava_train(transform_train, args.image_root, args.ann_root)
		val_dataset = cc3m_595k_llava_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val') 
		test_dataset = cc3m_595k_llava_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')         

	elif args.dataset == 'coyo':
		coyo_eval_max_samples = getattr(args, 'coyo_eval_max_samples', 10000)
		coyo_max_samples = getattr(args, 'coyo_max_samples', None)
		train_dataset = COYO700MSubset(
			transform_train,
			args.image_root,
			COYO700MSubsetConfig(max_samples=coyo_max_samples, split='train'),
		)
		val_dataset = COYO700MSubsetRetrievalEval(
			transform_test,
			args.image_root,
			COYO700MSubsetConfig(max_samples=coyo_eval_max_samples, split='val'),
		)
		test_dataset = COYO700MSubsetRetrievalEval(
			transform_test,
			args.image_root,
			COYO700MSubsetConfig(max_samples=coyo_eval_max_samples, split='test'),
		)

	elif args.dataset == 'mmcelebahq':
		train_dataset = mmcelebahq_train(transform_train, args.image_root, args.ann_root)
		val_dataset = mmcelebahq_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val')
		test_dataset = mmcelebahq_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')
			
	elif args.dataset=='flickr8k':		
		train_dataset = flickr8k_train(transform_train, args.image_root, args.ann_root)
		val_dataset = flickr8k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val') 
		test_dataset = flickr8k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')         
			
	# elif args.dataset=='pascal1k':
		# train_dataset = pascal1k_train(transform_train, args.image_root, args.ann_root)
		# val_dataset = pascal1k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val') 
		# test_dataset = pascal1k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')       
	else: 
		raise NotImplementedError

	samplers = [None, None, None]
	shuffles = [True, False, False]
	workers = [0] * len(samplers) # [min(8, os.cpu_count() or 4)] * len(samplers)
	batch_sizes = [args.batch_size_train, args.batch_size_test, args.batch_size_test]
	num_queries = getattr(args, 'num_queries', 100)
	batch_size_test = getattr(args, 'batch_size_test', 64)
	# batch_sizes = [num_queries, batch_size_train, batch_size_test]
	collate_fns = [None, None, None]
	pin_memory = [True, True, True]
	train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                       batch_size=batch_sizes,
                                                       num_workers=workers,
                                                       is_trains=shuffles, 
                                                       collate_fns=collate_fns,
                                                       pin_memory=pin_memory)

	return train_loader, test_loader, train_dataset, test_dataset


def get_dataset_flickr(args):
	"""Return train/test loaders and datasets (same interface as evaluate_only.py). Alias for get_dataloaders."""
	return get_dataloaders(args)


class PromptedClassificationDataset(Dataset):
	"""
	Wrap a classification dataset (image, label) to return (image, caption, index),
	where caption is a prompt-style string built from the class name.
	"""

	def __init__(self, base_dataset: Dataset, classes: List[str], prompt_template: str = "a photo of a {}."):
		self.base_dataset = base_dataset
		self.classes = list(classes)
		self.prompt_template = prompt_template

	def __len__(self) -> int:
		return len(self.base_dataset)

	def __getitem__(self, index: int):
		"""
		Robustly fetch (image, caption, label) from the underlying dataset.
		If an I/O error occurs when reading an image, retry a few nearby indices
		before giving up. This prevents a single corrupt file from crashing
		visualization or evaluation scripts.
		"""
		max_attempts = 5
		cur_idx = int(index)
		for _ in range(max_attempts):
			try:
				image, label = self.base_dataset[cur_idx]
				break
			except OSError:
				# Skip problematic sample and try the next one
				cur_idx = (cur_idx + 1) % len(self.base_dataset)
		else:
			# If all attempts fail, re-raise the last error
			image, label = self.base_dataset[index]

		class_name = self.classes[int(label)]
		caption = self.prompt_template.format(class_name)
		# Return both caption (for text encoder) and label (for classification metrics)
		return image, caption, int(label)


def create_zscls_dataset(args, min_scale: float = 0.5):
	"""
	Create zero-shot classification datasets (imagenet, cub, stcar, air, cifar100)
	from roots. Uses args.zscls_base_dir if set, else ZSCLS_BASE_DIR.
	BIA-style paths: see data/imagenet.py, data/cub.py, etc.

	Each train/test sample returns (image, caption, index), where caption is a
	text prompt of the form \"a photo of a {class}.\" suitable for BERT-style
	text encoders and distillation via get_images_texts.
	"""
	if args.dataset not in ZSCLS_DATASETS:
		raise ValueError(f"Dataset {args.dataset} is not a zscls dataset. Expected one of {sorted(ZSCLS_DATASETS)}.")

	base_dir = getattr(args, "zscls_base_dir", None) or ZSCLS_BASE_DIR
	paths = get_zscls_paths(args.dataset, base_dir)

	normalize = transforms.Normalize(
		(0.48145466, 0.4578275, 0.40821073),
		(0.26862954, 0.26130258, 0.27577711),
	)
	transform_train = transforms.Compose(
		[
			transforms.RandomResizedCrop(
				args.image_size,
				scale=(min_scale, 1.0),
				interpolation=InterpolationMode.BICUBIC,
			),
			transforms.RandomHorizontalFlip(),
			RandomAugment(
				2,
				5,
				isPIL=True,
				augs=[
					"Identity",
					"AutoContrast",
					"Brightness",
					"Sharpness",
					"Equalize",
					"ShearX",
					"ShearY",
					"TranslateX",
					"TranslateY",
					# "Rotate",
				],
			),
			transforms.ToTensor(),
			normalize,
		]
	)
	transform_test = transforms.Compose(
		[
			transforms.Resize(
				(args.image_size, args.image_size),
				interpolation=InterpolationMode.BICUBIC,
			),
			transforms.ToTensor(),
			normalize,
		]
	)

	ds = args.dataset

	def _get_zscls_prompt_template(ds_name: str) -> str:
		"""Load PROMPT_TEMPLATE from data.{ds_name}.py (e.g. data.imagenet, data.imagenet_r)."""
		mod_name = ds_name.replace("-", "_")
		try:
			mod = __import__(f"data.{mod_name}", fromlist=["PROMPT_TEMPLATE"])
			return getattr(mod, "PROMPT_TEMPLATE", "a photo of a {}.")
		except ImportError:
			return "a photo of a {}."

	prompt_template = _get_zscls_prompt_template(ds)

	if ds == "imagenet":
		train_root = paths.get("train_root") or os.path.join(base_dir, "imagenet", "train")
		val_root = paths.get("val_root") or os.path.join(base_dir, "imagenet", "val")
		base_train = tv_datasets.ImageFolder(train_root, transform=transform_train, allow_empty=True)
		base_test = tv_datasets.ImageFolder(val_root, transform=transform_test, allow_empty=True)
		synset_to_name = _load_imagenet_class_index()
		classes_train = [synset_to_name.get(c, c) for c in base_train.classes]
		classes_test = [synset_to_name.get(c, c) for c in base_test.classes]
		train_dataset = PromptedClassificationDataset(base_train, classes_train, prompt_template=prompt_template)
		test_dataset = PromptedClassificationDataset(base_test, classes_test, prompt_template=prompt_template)

	elif ds == "stcar":
		# Stanford Cars using annotations under /path/to/datasets/STCAR
		from PIL import Image  # local import to avoid global patch issues
		import csv

		class STCarsClassificationDataset(Dataset):
			def __init__(self, root: str, split: str, transform=None):
				self.root = os.path.expanduser(root)
				self.transform = transform
				if split not in {"train", "test"}:
					raise ValueError(f"Invalid split {split} for STCAR; expected 'train' or 'test'.")
				self.split = split

				anno_path = os.path.join(self.root, f"anno_{'train' if split == 'train' else 'test'}.csv")
				samples = []
				with open(anno_path, "r") as f:
					reader = csv.reader(f)
					for row in reader:
						if not row:
							continue
						filename = row[0]
						label = int(float(row[-1])) - 1  # last field is 1-based class index
						samples.append((filename, label))
				self.samples = samples

				names_path = os.path.join(self.root, "names.csv")
				with open(names_path, "r") as f:
					self.classes = [ln.strip() for ln in f if ln.strip()]

			def __len__(self) -> int:
				return len(self.samples)

			def __getitem__(self, idx: int):
				filename, label = self.samples[idx]
				img_dir = "cars_train" if self.split == "train" else "cars_test"
				img_path = os.path.join(self.root, img_dir, filename)
				image = Image.open(img_path).convert("RGB")
				if self.transform is not None:
					image = self.transform(image)
				return image, int(label)

		root = paths.get("image_root") or os.path.join(base_dir, "STCAR")
		base_train = STCarsClassificationDataset(root=root, split="train", transform=transform_train)
		base_test = STCarsClassificationDataset(root=root, split="test", transform=transform_test)
		classes_train = [_clean_zscls_class_name("stcar", c) for c in base_train.classes]
		classes_test = [_clean_zscls_class_name("stcar", c) for c in base_test.classes]
		train_dataset = PromptedClassificationDataset(base_train, classes_train, prompt_template=prompt_template)
		test_dataset = PromptedClassificationDataset(base_test, classes_test, prompt_template=prompt_template)

	elif ds == "air":
		# FGVC Aircraft classification using local annotation files, similar to AircraftDataset in BIA repo.
		from PIL import Image  # local import
		import numpy as _np  # avoid polluting global namespace

		class AirClassificationDataset(Dataset):
			def __init__(self, root: str, train: bool, transform=None):
				self.root = os.path.expanduser(root)
				self.transform = transform

				split = "trainval" if train else "test"
				classes_file = os.path.join(
					self.root,
					"fgvc-aircraft-2013b",
					"data",
					f"images_variant_{split}.txt",
				)
				img_folder = os.path.join(
					self.root,
					"fgvc-aircraft-2013b",
					"data",
					"images",
				)

				if not os.path.exists(classes_file):
					raise FileNotFoundError(f"AIR classes file not found: {classes_file}")

				image_ids: List[str] = []
				target_names: List[str] = []
				with open(classes_file, "r") as f:
					for line in f:
						line = line.strip()
						if not line:
							continue
						split_line = line.split(" ")
						image_ids.append(split_line[0])
						target_names.append(" ".join(split_line[1:]))

				classes = _np.unique(target_names)
				class_to_idx = {classes[i]: i for i in range(len(classes))}
				target_indices = [class_to_idx[c] for c in target_names]

				samples: List[Tuple[str, int]] = []
				for img_id, tgt in zip(image_ids, target_indices):
					img_path = os.path.join(img_folder, f"{img_id}.jpg")
					samples.append((img_path, int(tgt)))

				self.samples = samples
				self.classes = list(classes)

			def __len__(self) -> int:
				return len(self.samples)

			def __getitem__(self, idx: int):
				path, label = self.samples[idx]
				image = Image.open(path).convert("RGB")
				if self.transform is not None:
					image = self.transform(image)
				return image, int(label)

		root = paths.get("image_root") or os.path.join(base_dir, "AIR")
		base_train = AirClassificationDataset(root=root, train=True, transform=transform_train)
		base_test = AirClassificationDataset(root=root, train=False, transform=transform_test)
		classes_train = [_clean_zscls_class_name("air", c) for c in base_train.classes]
		classes_test = [_clean_zscls_class_name("air", c) for c in base_test.classes]
		train_dataset = PromptedClassificationDataset(base_train, classes_train, prompt_template=prompt_template)
		test_dataset = PromptedClassificationDataset(base_test, classes_test, prompt_template=prompt_template)

	elif ds == "cub":
		# CUB-200-2011 using official files under /path/to/datasets/CUB/CUB_200_2011/CUB_200_2011
		from PIL import Image  # local import
		import pandas as pd

		class CUBClassificationDataset(Dataset):
			def __init__(self, root: str, train: bool, transform=None):
				self.root = os.path.expanduser(root)
				self.transform = transform
				self.train = train

				images = pd.read_csv(
					os.path.join(self.root, "images.txt"),
					sep=" ",
					names=["img_id", "filepath"],
				)
				image_class_labels = pd.read_csv(
					os.path.join(self.root, "image_class_labels.txt"),
					sep=" ",
					names=["img_id", "target"],
				)
				train_test_split = pd.read_csv(
					os.path.join(self.root, "train_test_split.txt"),
					sep=" ",
					names=["img_id", "is_training_img"],
				)

				data = images.merge(image_class_labels, on="img_id")
				data = data.merge(train_test_split, on="img_id")
				if train:
					data = data[data.is_training_img == 1]
				else:
					data = data[data.is_training_img == 0]
				self.data = data.reset_index(drop=True)

				classes_df = pd.read_csv(
					os.path.join(self.root, "classes.txt"),
					sep=" ",
					names=["class_id", "class_name"],
				)
				self.classes = [row["class_name"] for _, row in classes_df.iterrows()]

			def __len__(self) -> int:
				return len(self.data)

			def __getitem__(self, idx: int):
				sample = self.data.iloc[idx]
				img_rel_path = sample.filepath
				img_path = os.path.join(self.root, "images", img_rel_path)
				image = Image.open(img_path).convert("RGB")
				if self.transform is not None:
					image = self.transform(image)
				label = int(sample.target) - 1
				return image, label

		root = paths.get("image_root") or os.path.join(base_dir, "CUB", "CUB_200_2011", "CUB_200_2011")
		base_train = CUBClassificationDataset(root=root, train=True, transform=transform_train)
		base_test = CUBClassificationDataset(root=root, train=False, transform=transform_test)
		classes_train = [_clean_zscls_class_name("cub", c) for c in base_train.classes]
		classes_test = [_clean_zscls_class_name("cub", c) for c in base_test.classes]
		train_dataset = PromptedClassificationDataset(base_train, classes_train, prompt_template=prompt_template)
		test_dataset = PromptedClassificationDataset(base_test, classes_test, prompt_template=prompt_template)

	elif ds == "cifar100":
		# CIFAR100 is 32x32; use resize 32 instead of default (e.g. 224)
		cifar_size = 32
		min_scale = 0.5
		transform_train_cifar = transforms.Compose(
			[
				transforms.RandomResizedCrop(
					cifar_size,
					scale=(min_scale, 1.0),
					interpolation=InterpolationMode.BICUBIC,
				),
				transforms.RandomHorizontalFlip(),
				RandomAugment(
					2,
					5,
					isPIL=True,
					augs=[
						"Identity",
						"AutoContrast",
						"Brightness",
						"Sharpness",
						"Equalize",
						"ShearX",
						"ShearY",
						"TranslateX",
						"TranslateY",
					],
				),
				transforms.ToTensor(),
				normalize,
			]
		)
		transform_test_cifar = transforms.Compose(
			[
				transforms.Resize(
					(cifar_size, cifar_size),
					interpolation=InterpolationMode.BICUBIC,
				),
				transforms.ToTensor(),
				normalize,
			]
		)
		root = paths.get("image_root") or os.path.join(base_dir, "cifar100")
		base_train = tv_datasets.CIFAR100(
			root=root,
			train=True,
			transform=transform_train_cifar,
			download=True,
		)
		base_test = tv_datasets.CIFAR100(
			root=root,
			train=False,
			transform=transform_test_cifar,
			download=True,
		)
		train_dataset = PromptedClassificationDataset(base_train, base_train.classes, prompt_template=prompt_template)
		test_dataset = PromptedClassificationDataset(base_test, base_test.classes, prompt_template=prompt_template)

	elif ds == "imagenet-r":
		# ImageNet-R: 200 classes (synset IDs), eval-only; use same root for train/test.
		root = paths.get("image_root") or os.path.join(base_dir, "imagenet-r")
		base = tv_datasets.ImageFolder(root, transform=transform_test)
		synset_to_name = _load_imagenet_class_index()
		classes = [synset_to_name.get(c, c) for c in base.classes]
		train_dataset = PromptedClassificationDataset(base, classes, prompt_template=prompt_template)
		test_dataset = PromptedClassificationDataset(base, classes, prompt_template=prompt_template)

	elif ds == "imagenet-a":
		# ImageNet-A: 200 ImageNet classes, eval-only; use same root for train/test.
		root = paths.get("image_root") or os.path.join(base_dir, "imagenet-a")
		base = tv_datasets.ImageFolder(root, transform=transform_test)
		synset_to_name = _load_imagenet_class_index()
		classes = [synset_to_name.get(c, c) for c in base.classes]
		train_dataset = PromptedClassificationDataset(base, classes, prompt_template=prompt_template)
		test_dataset = PromptedClassificationDataset(base, classes, prompt_template=prompt_template)

	else:
		raise NotImplementedError(f"ZSCLS dataset {ds} not implemented.")

	return train_dataset, test_dataset


def get_zscls_dataloaders(args):
	"""
	Return train/test dataloaders and datasets for zero-shot classification datasets.
	Interface mirrors get_dataloaders but only supports ZSCLS_DATASETS.
	"""
	train_dataset, test_dataset = create_zscls_dataset(args)

	samplers = [None, None]
	shuffles = [True, False]
	workers = [0, 0]
	batch_sizes = [args.batch_size_train, args.batch_size_test]
	collate_fns = [None, None]
	pin_memory = [True, True]

	train_loader, test_loader = create_loader(
		[train_dataset, test_dataset],
		samplers,
		batch_size=batch_sizes,
		num_workers=workers,
		is_trains=shuffles,
		collate_fns=collate_fns,
		pin_memory=pin_memory,
	)

	return train_loader, test_loader, train_dataset, test_dataset




# def load_synthetic_data(args, train_loader, fabric):
# 	"""Load or create synthetic data for evaluation."""
# 	import time
# 	start_time = time.time()
	
# 	device = fabric.device if fabric is not None else 'cuda' if torch.cuda.device_count() > 1 else 'cpu'
	
# 	NUM_SYN = args.num_queries
# 	IMAGE_SIZE = args.image_size
# 	TEXT_DIM = args.text_dim

# 	image_syn, text_syn = None, None
# 	assert args.pix_init == args.txt_init, "Image and text initialization must be the same"

# 	if args.pix_init == 'real' and args.txt_init == 'real':
# 		SYNTHETIC_DATA_TYPE = 'real'
# 		from src.vl_distill_utils import get_images_texts
# 		image_syn, text_syn = get_images_texts(NUM_SYN, train_loader.dataset, args)
# 		print(f"Loading synthetic data from [green]real[/green] data...")
		
# 	elif args.pix_init == 'noise' and args.txt_init == 'noise':
# 		SYNTHETIC_DATA_TYPE = 'noise'
	
# 		mean = torch.tensor([-0.0626, -0.0221,  0.0680])
# 		std  = torch.tensor([1.0451, 1.0752, 1.0539])
# 		image_syn = torch.randn([NUM_SYN, 3, IMAGE_SIZE, IMAGE_SIZE])
# 		for c in range(3):
# 			image_syn[:, c] = image_syn[:, c] * std[c] + mean[c]
# 		text_syn = torch.normal(mean=-0.0094, std=0.5253, size=(NUM_SYN, TEXT_DIM))
# 		print(f"Initializing with [green]noise[/green] synthetic data...")
		
	
# 	elif SYNTHETIC_DATA_TYPE == 'herding':
# 		image_syn, text_syn = coreset(method='herding', dataset=train_loader.dataset, num_syn=NUM_SYN, args=args)
# 		print(f"Initializing with [green]herding[/green] synthetic data...")
		
# 	elif SYNTHETIC_DATA_TYPE == 'kcenter':
# 		image_syn, text_syn = coreset(method='kcenter', dataset=train_loader.dataset, num_syn=NUM_SYN, args=args)
# 		print(f"Initializing with [green]kcenter[/green] synthetic data...")
		
# 	elif SYNTHETIC_DATA_TYPE == 'coreset':
# 		image_syn, text_syn = coreset(method='coreset', dataset=train_loader.dataset, num_syn=NUM_SYN, args=args)
# 		print(f"Initializing with [green]coreset[/green] synthetic data...")
# 		return image_syn, text_syn
# 	else:
# 		raise NotImplementedError(f"Synthetic data type {SYNTHETIC_DATA_TYPE} not implemented")
	
# 	assert SYNTHETIC_DATA_TYPE in ['real', 'random', 'herding', 'kcenter', 'noise'], f"Synthetic data type {SYNTHETIC_DATA_TYPE} not implemented"
	
# 	if image_syn is None or text_syn is None:
# 		raise ValueError(f"Failed to create synthetic data of type {SYNTHETIC_DATA_TYPE}")
	
# 	# Set requires_grad for optimization
# 	image_syn = image_syn.detach().to(device).requires_grad_(True)
# 	text_syn = text_syn.detach().to(device).requires_grad_(True)
	
# 	end_time = time.time()
# 	print(f"Synthetic data created in [green]{end_time - start_time:.2f} sec[/green]")
# 	print(f"Image: [green]{image_syn.shape}[/green], Text: [green]{text_syn.shape}[/green]")
	
		
# 	return image_syn, text_syn


class SimilarityDataLoaderWrapper:
	"""make a regular dataloader looks like a SimilarityDataloader,
	by return N samples and a N*N identity similarity matrix together"""

	def __init__(self, dataloader):
		super().__init__()
		self.dataloader = dataloader

	def __iter__(self):
		for data in self.dataloader:
			bz = data[0].shape[0]
			yield data + [torch.eye(bz, dtype=torch.float32).to(data[0].device)]


class SimilarityDataloader:
	def __init__(self, images_train, labels_train, similarity_train, batch_size, drop_last=False):
		"""images_train, labels_train: N samples
		similarity_train: N*N similarity matrix, similarity_train[i,j] is the similarity between i-th and j-th samples"""
		super().__init__()

		self.images_train     = images_train
		self.labels_train     = labels_train
		self.similarity_train = similarity_train
		self.batch_size       = batch_size
		assert not drop_last

	def __iter__(self):
		size = self.images_train.shape[0]
		num_batch = int(ceil(size / self.batch_size))
		indices = np.arange(size)
		for _ in range(num_batch):
			np.random.shuffle(indices)
			ids = indices[:self.batch_size]
			yield self.images_train[ids], self.labels_train[ids], self.similarity_train[ids[:,None], ids]




class MakeSyntheticDataloader:
	def __init__(self, images_train, labels_train, batch_size, drop_last=False):
		"""images_train, labels_train: N samples"""
		super().__init__()

		self.images_train = images_train
		self.labels_train = labels_train
		self.batch_size   = batch_size
		assert not drop_last

	def __iter__(self):
		size = self.images_train.shape[0]
		num_batch = int(ceil(size / self.batch_size))
		indices = np.arange(size)
		for _ in range(num_batch):
			np.random.shuffle(indices)
			ids = indices[:self.batch_size]
			yield self.images_train[ids], self.labels_train[ids]
