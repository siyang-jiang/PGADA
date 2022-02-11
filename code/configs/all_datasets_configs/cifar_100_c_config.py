from pathlib import Path

from src.data_tools.datasets import CIFAR100CMeta

DATASET = CIFAR100CMeta
IMAGE_SIZE = 32
DATA_ROOT = Path("../data")
SPECS_ROOT = Path("configs/dataset_specs/cifar_100_c")
CLASSES = {"train": 65, "val": 10, "test": 25}
