"""
Zero-shot classification: FGVC Aircraft.
BIA-style: rawdata_root = AIR/fgvc-aircraft-2013b/data/images; anno_root has train.txt, test.txt.
"""
import os

ZSCLS_BASE_DIR = os.environ.get("ZSCLS_BASE_DIR", "/path/to/datasets")
IMAGE_ROOT = os.path.join(ZSCLS_BASE_DIR, "AIR")  # HGA: root/fgvc-aircraft-2013b/data/...

PROMPT_TEMPLATE = "a photo of a {}, a type of aircraft."
PROMPT_TEMPLATES = [
    "a photo of a {}, a type of aircraft.",
    "a photo of the {}, a type of aircraft.",
]


def get_paths(base_dir: str) -> dict:
    """BIA-style: root = AIR (HGA uses root/fgvc-aircraft-2013b/data/...); anno_root for train.txt/test.txt."""
    image_root = os.path.join(base_dir, "AIR")  # HGA expects AIR root
    anno_root = os.path.join(base_dir, "AIR_annotations")
    return {
        "image_root": image_root,
        "ann_root": anno_root,
    }
