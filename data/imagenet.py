"""
Zero-shot classification: ImageNet.
Paths and CLIP-style prompts for ImageNet (ILSVRC).
BIA-style: ImageFolder with train/ and val/ subdirs under base.
"""
import os

# Default base dir for ImageNet (train/val under this)
ZSCLS_BASE_DIR = os.environ.get("ZSCLS_BASE_DIR", "/path/to/datasets")
IMAGE_ROOT = os.path.join(ZSCLS_BASE_DIR, "imagenet")
TRAIN_SUBDIR = "train"
VAL_SUBDIR = "val"
# ImageFolder: no separate anno; classes from folder names

PROMPT_TEMPLATE = "a photo of a {}."
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
    "a photo of the {}.",
]


def get_paths(base_dir: str) -> dict:
    """Return paths for BIA-style loading. base_dir = DATA_ROOT_DIR."""
    root = os.path.join(base_dir, "imagenet")
    return {
        "image_root": root,
        "train_root": os.path.join(root, "train"),
        "val_root": os.path.join(root, "val"),
        "ann_root": None,
    }
