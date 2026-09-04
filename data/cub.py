"""
Zero-shot classification: CUB-200-2011 (birds).
BIA-style: rawdata_root = CUB/CUB_200_2011/CUB_200_2011/images; anno_root has train.txt, test.txt.
"""
import os

ZSCLS_BASE_DIR = os.environ.get("ZSCLS_BASE_DIR", "/path/to/datasets")
IMAGE_ROOT = os.path.join(ZSCLS_BASE_DIR, "CUB", "CUB_200_2011", "CUB_200_2011")
# BIA DCL: rawdata_root = .../images; anno_root has train.txt, test.txt (format: "ImageName label")
ANNO_SUBDIR = "CUB"  # relative to project, or use full path under base

PROMPT_TEMPLATE = "a photo of a {}."
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of a bird, {}.",
    "a photo of the {}.",
]


def get_paths(base_dir: str) -> dict:
    """BIA-style: rawdata_root (CUB_200_2011); anno_root for train.txt/test.txt if using BIA format."""
    rawdata_root = os.path.join(base_dir, "CUB", "CUB_200_2011", "CUB_200_2011")
    anno_root = os.path.join(base_dir, "CUB_annotations")  # train.txt, test.txt
    return {
        "image_root": rawdata_root,  # HGA uses root with images/ subdir
        "rawdata_root": rawdata_root,
        "ann_root": anno_root,
    }
