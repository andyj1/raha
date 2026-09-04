"""
Zero-shot classification: CIFAR-100.
BIA-style: torchvision CIFAR100 under base_dir/cifar100.
"""
import os

ZSCLS_BASE_DIR = os.environ.get("ZSCLS_BASE_DIR", "/path/to/datasets")
IMAGE_ROOT = os.path.join(ZSCLS_BASE_DIR, "cifar100")

PROMPT_TEMPLATE = "a photo of a {}."
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
]


def get_paths(base_dir: str) -> dict:
    """CIFAR-100: root for torchvision CIFAR100."""
    return {
        "image_root": os.path.join(base_dir, "cifar100"),
        "ann_root": None,
    }
