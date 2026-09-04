"""Move some basic utils in distill.py in VL-Distill here"""
import os
import numpy as np
import copy
import torch
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from src.networks import TextEncoder, ImageEncoder
from tqdm import tqdm

import clip


def get_images_texts(n, dataset, args, text_encoder, seed=None, get_text_raw=False, init='random'):
    
    if init == 'random':
        if seed != None:
            np.random.seed(seed)
        idx_shuffle = np.random.permutation(len(dataset))[:n]

        # Initialize the text encoder
        with torch.no_grad():
            text_encoder.eval()

            image_syn = torch.stack([dataset[i][0] for i in idx_shuffle])
            texts = [dataset[i][1] for i in idx_shuffle]

            encoding = text_encoder.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(args.device)
            attention_mask = encoding['attention_mask'].to(args.device)

            text_syn = text_encoder.model.embeddings(
                input_ids=input_ids,
            )
    elif init == 'noise':
        mean = torch.tensor([-0.0626, -0.0221,  0.0680])
        std  = torch.tensor([1.0451, 1.0752, 1.0539])
        attention_mask = torch.ones([args.num_queries, 1, 1, 1])
        image_syn = torch.randn([args.num_queries, 3, 224, 224])
        for c in range(3):
            image_syn[:, c] = image_syn[:, c] * std[c] + mean[c]
        text_syn = torch.normal(mean=-0.0094, std=0.5253, size=(args.num_queries, 768))
        texts = None
    else:
        raise NotImplementedError(f"Initialization method {init} not implemented")
    
    if get_text_raw:
        return image_syn, text_syn.float(), attention_mask, texts
    else:
        return image_syn, text_syn.float(), attention_mask

