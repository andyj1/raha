"""
Zero-shot classification: Stanford Cars.
BIA-style: rawdata_root = STCAR; anno_root has train.txt, test.txt (format: "ImageName label").
"""
import os

ZSCLS_BASE_DIR = os.environ.get("ZSCLS_BASE_DIR", "/path/to/datasets")
IMAGE_ROOT = os.path.join(ZSCLS_BASE_DIR, "STCAR")

PROMPT_TEMPLATE = "a photo of a {}."
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of my {}.",
    "a photo of my clean {}.",
    "a photo of my old {}.",
]


def get_paths(base_dir: str) -> dict:
    """BIA-style: rawdata_root for images, anno_root for train.txt/test.txt."""
    image_root = os.path.join(base_dir, "STCAR")
    anno_root = os.path.join(base_dir, "STCAR_annotations")
    return {
        "image_root": image_root,
        "ann_root": anno_root,
    }
