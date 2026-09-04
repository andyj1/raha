from functools import lru_cache
import os
from tqdm import tqdm
import yaml
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import re
import json
from PIL import Image

__all__ = ['flickr8k_train', 'flickr8k_retrieval_eval']

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


class flickr8k_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='', download=True):        
        '''
        Modified to use Karpathy format annotation files from /path/to/annotations_retrieval/
        '''        
        urls = {'train': 'https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr8k_train_karpathy.txt'}
        filename = {'train':'flickr8k_train_karpathy.txt'}
        
        if download:
            download_url(urls['train'], ann_root, filename['train'])
            
        self.transform = transform
        self.image_root = os.path.join(image_root, 'flickr8k-images')
        
        self.max_words = max_words      
        self.prompt = prompt
        
        # Load Karpathy train annotation file
        train_ann_file = os.path.join(ann_root, filename['train'])

        # Parse CSV format: image,caption
        self.annotation = []
        self.img_ids = {}
        n = 0
        
        with open(train_ann_file, 'r') as f:
            lines = f.readlines()
            # Skip header line
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse CSV format: image,caption
                parts = line.split(',', 1)  # Split only on first comma
                if len(parts) == 2:
                    img_filename, caption = parts
                    
                    self.annotation.append({
                        'image': img_filename,
                        'caption': caption,
                        'image_id': img_filename
                    })
                    
                    if img_filename not in self.img_ids:
                        self.img_ids[img_filename] = n
                        n += 1
        
    def __len__(self):
        return len(self.annotation)
    
    @lru_cache(maxsize=100)
    def read_image(self, image_path):      
        try:
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
        except:
            pass
        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        return image
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])
        image = self.read_image(image_path)      
        
        caption = self.prompt + pre_caption(ann['caption'], self.max_words) 

        # return image, caption, self.img_ids[ann['image_id']], index
        return image, caption, index
        
    def get_all_captions(self):
        captions = []
        for ann in self.annotation:
            caption = self.prompt + pre_caption(ann['caption'], self.max_words)
            captions.append(caption)
        return captions

    
    
class flickr8k_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split='test', max_words=30, download=True):  
        '''
        Modified to use Karpathy format annotation files from /path/to/annotations_retrieval/
        '''
        urls = {'val': 'https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr8k_val_karpathy.txt',
                'test': 'https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr8k_test_karpathy.txt'}
        filenames = {'val':'flickr8k_val_karpathy.txt', 
                     'test':'flickr8k_test_karpathy.txt'}
        
        if download:
            download_url(urls[split],ann_root, filenames[split])
            
        self.transform = transform
        self.image_root = os.path.join(image_root, 'flickr8k-images')
        self.max_words = max_words   
        
        # Load Karpathy test annotation file
        test_ann_file = os.path.join(ann_root, filenames[split])
        
        # Parse CSV format and group by image
        image_captions = {}
        
        with open(test_ann_file, 'r') as f:
            lines = f.readlines()
            # Skip header line
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse CSV format: image,caption
                parts = line.split(',', 1)  # Split only on first comma
                if len(parts) == 2:
                    img_filename, caption = parts
                    
                    if img_filename not in image_captions:
                        image_captions[img_filename] = []
                    image_captions[img_filename].append(caption)
        
        # Create annotation list for evaluation
        self.annotation = []
        for img_filename, captions in image_captions.items():
            self.annotation.append({
                'image': img_filename,
                'caption': captions  # All captions for this image
            })
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for caption in ann['caption']:
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])       
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index
