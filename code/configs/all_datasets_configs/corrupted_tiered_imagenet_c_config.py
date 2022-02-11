from functools import partial
from pathlib import Path

from src.data_tools.datasets import TieredImageNetC

"""
Config for tieredImageNet-C where all corrupted images are explicitly written on disk.
"""

DATASET = partial(TieredImageNetC, load_corrupted_dataset=True)
DATA_ROOT = Path("")
IMAGE_SIZE = 224
SPECS_ROOT = Path("configs/dataset_specs/tiered_imagenet_c")
CLASSES = {"train": 351, "val": 97, "test": 160}
