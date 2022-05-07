import json
from functools import partial
from pathlib import Path

import os
import torch
from PIL import Image
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm

from configs.dataset_specs.tiered_imagenet_c.perturbation_params import (
    PERTURBATION_PARAMS,
)
from src.data_tools.samplers import AfterCorruptionSampler, BeforeCorruptionSampler
from src.data_tools.transform import TransformLoader
from src.data_tools.utils import get_perturbations, load_image_as_array


class MiniImageNetC(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        target_transform: Optional[Callable] = None,
        augmentation: Optional[Callable] = False,
        two_stream: Optional[Callable] = False,
        SIMCLR_val: bool = False
    ):
        self.two_stream = two_stream
        self.split = split
        self.SIMCLR_val = SIMCLR_val
        transform = TransformLoader(image_size).get_composed_transform(aug=augmentation)
        self.transform_test = TransformLoader(image_size).get_composed_transform(aug=False)

        super(MiniImageNetC, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        # We need to write this import here (and not at the top) to avoid cyclic imports
        from configs.dataset_config import SPECS_ROOT

        # Get perturbations
        with open(SPECS_ROOT / f"{split}.json", "r") as file:
            split_specs = json.load(file)
        self.perturbations, self.id_to_domain = get_perturbations(
            split_specs["perturbations"], PERTURBATION_PARAMS, image_size
        )
        self.domain_to_id = {v: k for k, v in self.id_to_domain.items()}

        # Get images and labels
        images_df = pd.read_csv(SPECS_ROOT / f"{split}_images.csv").assign(
            image_paths=lambda df: df.apply(
                lambda row: os.path.join(root, *row), axis=1
            )
        )
        self.images = np.stack(
            [
                load_image_as_array(image_path, image_size)
                for image_path in tqdm(images_df.image_paths)
            ]
        )

        self.class_list = images_df.class_name.unique()
        self.id_to_class = dict(enumerate(self.class_list))
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
        self.labels = list(images_df.class_name.map(self.class_to_id))

    def __len__(self):
        return len(self.images) * len(self.perturbations)

    def __getitem__(self, item):
        original_data_index = item // len(self.perturbations)
        perturbation_index = item % len(self.perturbations)

        img, label = (
            Image.fromarray(self.images[original_data_index]),
            self.labels[original_data_index],
        )

        img_p = self.perturbations[perturbation_index](img)

        if self.transform is not None:
            # TODO: some perturbations output arrays, some output images. We need to clean that.
            if isinstance(img_p, np.ndarray):
                img_p = img_p.astype(np.uint8)
                img_p = Image.fromarray(img_p)
            img_p = self.transform(img_p)
            img1 = self.transform_test(img) if self.SIMCLR_val else self.transform(img)
            img2 = self.transform(img)
            

        if self.target_transform is not None:
            label = self.target_transform(label)
        
        if self.two_stream:
            if self.split == "test":
                return img1, img2, label, perturbation_index
            else:
                return img1, img_p, label, perturbation_index
        else:
            return img_p, label, perturbation_index

    def get_sampler(self):
        return partial(BeforeCorruptionSampler, self)
