import os
import json
from functools import lru_cache
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
import re
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np

__all__ = ['cc3m_595k_llava_train', 'cc3m_595k_llava_retrieval_eval']

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

class cc3m_595k_llava_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        # url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'cc3m_595k_llava_train.json'

        # download_url(url,ann_root)
        
        self.annotation_filepath = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.annotation = []
        self.img_ids = {}  
        
        n = 0
        for ann in self.annotation_filepath:
            img_filename = ann['image']
            caption = ann['caption']
            # Each annotation should have: image filename, caption, image_id (the filename for uniqueness)
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
        
        image_path = os.path.join(self.image_root, 'images', ann['image'])
        image = self.read_image(image_path)     
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']], index
    
    
    def get_all_captions(self):
        captions = []
        for ann in self.annotation:
            caption = self.prompt + pre_caption(ann['caption'], self.max_words)
            captions.append(caption)
        return captions

    
class cc3m_595k_llava_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30, prompt=''):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        # urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                # 'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'cc3m_595k_llava_val.json',
                     'test':'cc3m_595k_llava_test.json'}

        # download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.annotation = self.annotation[:10000]
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            # for i, caption in enumerate(ann['caption']):
            caption = ann['caption']
            self.text.append(pre_caption(caption,max_words))
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, 'images', self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index


def visualize_dataset_samples(dataset, num_samples=5, save_path='cc3m_verified.png'):
    """
    Visualize dataset samples with images and captions side by side.
    
    Args:
        dataset: The dataset to visualize
        num_samples: Number of samples to show
        save_path: Path to save the visualization
    """
    # Create a transform for visualization (no normalization)
    vis_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Create a temporary dataset with visualization transform
    if hasattr(dataset, 'transform'):
        original_transform = dataset.transform
        dataset.transform = vis_transform
    
    # Create figure with better spacing and layout
    fig, axes = plt.subplots(num_samples, 2, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Adjust subplot spacing to prevent overlap
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    random_indices = np.random.permutation(len(dataset))
    for i in range(num_samples):
        # Select a random sample from random permutation of indices
        random_index = random_indices[i]
        
        try:
            # Get a sample - handle different dataset types
            if hasattr(dataset, 'img_ids'):  # Training dataset
                img, caption, img_id, index = dataset[random_index]
            else:  # Evaluation dataset
                img, index = dataset[random_index]
                caption = f"Sample {i+1}"  # Placeholder caption for eval datasets
            
            # Convert tensor back to PIL for display
            if isinstance(img, torch.Tensor):
                # Denormalize if needed
                if img.min() < 0:  # Check if normalized
                    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = img.clamp(0, 1)
                img = transforms.ToPILImage()(img)
            
            # Display image with proper aspect ratio
            axes[i, 0].imshow(img, aspect='equal')
            axes[i, 0].set_title(f'Image {i+1}', fontsize=12, fontweight='bold', pad=10)
            axes[i, 0].axis('off')
            
            # Display caption with better text wrapping and positioning
            # Wrap long captions
            max_chars_per_line = 50
            if len(caption) > max_chars_per_line:
                words = caption.split()
                wrapped_lines = []
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= max_chars_per_line:
                        current_line += (" " + word) if current_line else word
                    else:
                        if current_line:
                            wrapped_lines.append(current_line)
                        current_line = word
                if current_line:
                    wrapped_lines.append(current_line)
                wrapped_caption = "\n".join(wrapped_lines)
            else:
                wrapped_caption = caption
            
            # Clear the text axis and set up proper boundaries
            axes[i, 1].clear()
            axes[i, 1].set_xlim(0, 1)
            axes[i, 1].set_ylim(0, 1)
            
            # Position text with proper margins
            axes[i, 1].text(0.05, 0.5, wrapped_caption, 
                           fontsize=11, 
                           verticalalignment='center',
                           horizontalalignment='left',
                           bbox=dict(boxstyle="round,pad=0.5", 
                                    facecolor="lightblue", 
                                    alpha=0.8,
                                    edgecolor="navy",
                                    linewidth=1),
                           wrap=True)
            axes[i, 1].set_title(f'Caption {i+1}', fontsize=12, fontweight='bold', pad=10)
            axes[i, 1].axis('off')
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            # Display error message
            axes[i, 0].text(0.5, 0.5, f'Error loading image {i+1}', 
                           ha='center', va='center', fontsize=10)
            axes[i, 0].set_title(f'Image {i+1} (Error)', fontsize=10)
            axes[i, 0].axis('off')
            
            axes[i, 1].text(0.1, 0.5, f'Error: {str(e)[:50]}...', fontsize=8, 
                           verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            axes[i, 1].set_title(f'Caption {i+1} (Error)', fontsize=10)
            axes[i, 1].set_xlim(0, 1)
            axes[i, 1].set_ylim(0, 1)
            axes[i, 1].axis('off')
    
    # Use tight_layout with padding to prevent overlap
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {save_path}")
    plt.close()
    
    # Restore original transform
    if hasattr(dataset, 'transform'):
        dataset.transform = original_transform


def test_cc3m_dataset():
    """Test function for CC3M dataset"""
    print("Testing CC3M Dataset...")

    from torch.utils.data import DataLoader

    # Create transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Paths
    image_root = '/mnt/sdc/datasets/cc3m_595k_llava'
    ann_root = '/mnt/sdc/datasets/annotations_retrieval'

    if not os.path.exists(image_root):
        print(f"❌ CC3M images path does not exist: {image_root}")
        print("   Please ensure the CC3M dataset is available at the specified path.")
        return False

    if not os.path.exists(ann_root):
        print(f"❌ CC3M annotations path does not exist: {ann_root}")
        print("   Please ensure the CC3M dataset is available at the specified path.")
        return False

    # --- Training dataset test ---
    print("1. Testing CC3M training dataset...")
    try:
        train_dataset = cc3m_595k_llava_train(
            transform=transform,
            image_root=image_root,
            ann_root=ann_root,
            max_words=30,
            prompt='',
        )
        print(f"   - Dataset length: {len(train_dataset)}")
    except Exception as e:
        print(f"   - ⚠️ Error loading training dataset: {e}")
        return False

    # Try using a DataLoader to iterate a few batches
    try:
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        for batch in train_loader:
            imgs, captions, *_ = batch
            print(f"   - Loaded batch from train dataset: imgs shape {imgs.shape}, batch size {imgs.shape[0]}")
            print(f"   - Example caption: {captions[0]}")
            break
    except Exception as e:
        print(f"   - ⚠️ DataLoader error (train): {e}")
        return False

    # Test get_all_captions
    try:
        all_captions = train_dataset.get_all_captions()
        print(f"   - Number of captions: {len(all_captions)}")
    except Exception as e:
        print(f"   - ⚠️ Error calling get_all_captions: {e}")
        return False

    # --- Retrieval eval dataset test ---
    print("2. Testing CC3M retrieval eval dataset...")

    try:
        eval_dataset = cc3m_595k_llava_retrieval_eval(
            transform=transform,
            image_root=image_root,
            ann_root=ann_root,
            split='val',
            max_words=30,
        )
        print(f"   - Dataset length: {len(eval_dataset)}")
        print(f"   - Number of text entries: {len(eval_dataset.text)}")
    except Exception as e:
        print(f"   - ⚠️ Error loading retrieval eval dataset: {e}")
        return False

    # Try using a DataLoader to iterate a few batches for eval set
    try:
        eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=True)
        for batch in eval_loader:
            imgs, img_ids = batch
            print(f"   - Loaded batch from eval dataset: imgs shape {imgs.shape}, batch size {imgs.shape[0]}")
            print(f"   - Example img_id: {img_ids[0]}")
            break
    except Exception as e:
        print(f"   - ⚠️ DataLoader error (eval): {e}")
        return False

    
    # Create visualization
    print("3. Creating visualization...")
    if len(train_dataset) > 0:
        visualize_dataset_samples(train_dataset, num_samples=min(5, len(train_dataset)), save_path='cc3m_595k_llava_verified.png')
    else:
        print("   - Skipping visualization: no evaluation samples available")
    
    print("✅ CC3M 595K LLava dataset test passed!")
    return True

if __name__ == "__main__":
    test_cc3m_dataset()
