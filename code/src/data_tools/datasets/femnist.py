from functools import partial
from pathlib import Path

from typing import Callable, Optional

import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision import transforms

from src.data_tools.samplers import GroupedDatasetSampler

from PIL import Image

from src.data_tools.transform import TransformLoader

class FEMNIST(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 28,
        target_transform: Optional[Callable] = None,
        augmentation: Optional[Callable] = False,
        two_stream = False,
        SIMCLR_val = False
    ):
        self.two_stream = two_stream
        self.SIMCLR_val = SIMCLR_val
        transform = TransformLoader(image_size).get_composed_transform(aug=augmentation)
        self.transform_test = TransformLoader(image_size).get_composed_transform(aug=False)

        super(FEMNIST, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.root = Path(root)

        self.images = np.load(self.root / f"{split}.npy")

        # We need to write this import here (and not at the top) to avoid cyclic imports
        from configs.dataset_config import SPECS_ROOT

        self.meta_data = pd.read_csv(SPECS_ROOT / f"{split}.csv", index_col=0)

        self.id_to_class = dict(
            enumerate(self.meta_data.class_name.sort_values().unique())
        )
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
        self.id_to_domain = dict(enumerate(self.meta_data.user.unique()))
        self.domain_to_id = {v: k for k, v in self.id_to_domain.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        label = self.class_to_id[self.meta_data.class_name.iloc[int(item)]]
        domain_id = self.domain_to_id[self.meta_data.user.iloc[int(item)]]

        img = transforms.ToTensor()(self.images[int(item)]).repeat(3, 1, 1).float()
        img = transforms.ToPILImage()(img).convert('RGB')
        # img = self.transform(img)
        # if self.transform is not None:
        #     img = self.images[int(item)]
        #     # TODO: some perturbations output arrays, some output images. We need to clean that.
        #     if isinstance(img, np.ndarray):
        #         # img = img.astype(np.uint8)
        #         img = Image.fromarray(img, 'RGB')
        #     img = self.transform(img)
        # else:
        #     img = self.transform(self.images[int(item)]).repeat(3, 1, 1).float()

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.two_stream:
            if self.SIMCLR_val:
                return self.transform_test(img), self.transform(img), label, domain_id
            else:
                return self.transform(img), self.transform(img), label, domain_id
        else:
            return self.transform_test(img), label, domain_id

    def get_sampler(self):
        return partial(GroupedDatasetSampler, self)
