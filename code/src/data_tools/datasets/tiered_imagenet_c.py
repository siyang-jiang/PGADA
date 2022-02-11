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
from src.data_tools.utils import get_perturbations


class TieredImageNetC(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        target_transform: Optional[Callable] = None,
        load_corrupted_dataset=False,
    ):
        transform = TransformLoader(image_size).get_composed_transform(aug=False)
        super(TieredImageNetC, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        # We need to write this import here (and not at the top) to avoid cyclic imports
        from configs.dataset_config import SPECS_ROOT

        with open(SPECS_ROOT / f"{split}.json", "r") as file:
            split_specs = json.load(file)

        self.root = Path(root)
        self.class_list = split_specs["class_names"]
        self.id_to_class = dict(enumerate(self.class_list))
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}

        self.perturbations, self.id_to_domain = get_perturbations(
            split_specs["perturbations"], PERTURBATION_PARAMS, image_size
        )
        self.domain_to_id = {v: k for k, v in self.id_to_domain.items()}

        self.load_corrupted_dataset = load_corrupted_dataset
        logger.info(f"Retrieving {split} images ...")
        if self.load_corrupted_dataset:
            pickle_path = self.root / f"{split}.pkl"
            if pickle_path.exists():
                self.images_df = pd.read_pickle(pickle_path)
            else:
                self.images_df = (
                    pd.concat(
                        [
                            pd.DataFrame(
                                [
                                    [
                                        img_path.parts[-1],
                                        self.domain_to_id[img_path.parts[-2]],
                                    ]
                                    for img_path in (self.root / class_name).glob(
                                        "*/*.png"
                                    )
                                ],
                                columns=["img_name", "domain_id"],
                            ).assign(class_id=class_id)
                            for class_name, class_id in tqdm(
                                self.class_to_id.items(), unit="classes"
                            )
                        ],
                        ignore_index=True,
                    )
                    .sort_values(by=["class_id", "img_name", "domain_id"])
                    .reset_index(drop=True)
                )
                self.images_df.to_pickle(pickle_path)
        else:
            self.images, self.labels = self.get_images_and_labels()
            self.images_df = pd.DataFrame(
                {
                    "class_id": self.labels,
                    "img_name": [os.path.basename(x) for x in self.images],
                    "key": 1,
                }
            ).merge(
                pd.DataFrame({"domain_id": list(self.id_to_domain.keys()), "key": 1}),
                on="key",
            )[
                ["class_id", "domain_id", "img_name"]
            ]

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, item):
        img_name = self.images_df.img_name.iloc[int(item)]
        label = self.images_df.class_id.iloc[int(item)]
        domain_id = self.images_df.domain_id.iloc[int(item)]

        if self.load_corrupted_dataset:
            img = self.transform(
                Image.open(
                    self.root
                    / self.id_to_class[label]
                    / self.id_to_domain[domain_id]
                    / img_name
                )
            )
        else:
            img = Image.open(self.root / self.id_to_class[label] / img_name).convert(
                "RGB"
            )
            img = transforms.Resize((224, 224))(img)
            img = self.perturbations[domain_id](img)
            if isinstance(img, np.ndarray):
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
            img = transforms.ToTensor()(img).type(torch.float32)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, domain_id

    def get_images_and_labels(self):
        """
        Provides image paths and corresponding labels, as expected to define our VisionDataset objects.
        Returns:
            tuple(list(str), list(int): respectively the list of all paths to images belonging in the split defined in
            the input JSON file, and their class ids
        """

        image_names = []
        image_labels = []

        for class_id, class_name in enumerate(self.class_list):
            class_images_paths = [
                str(image_path)
                for image_path in (self.root / class_name).glob("*")
                if image_path.is_file()
            ]
            image_names += class_images_paths
            image_labels += len(class_images_paths) * [class_id]

        return image_names, image_labels

    def get_sampler(self):
        if self.load_corrupted_dataset:
            return partial(AfterCorruptionSampler, self)
        else:
            return partial(BeforeCorruptionSampler, self)
