from PIL import ImageEnhance

import torch
from torchvision import transforms as transforms

TRANSFORM_TYPES = dict(
    Brightness=ImageEnhance.Brightness,
    Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness,
    Color=ImageEnhance.Color,
)

NORMALIZE_DEFAULT = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

JITTER_DEFAULT = dict(Brightness=0.4, Contrast=0.4, Color=0.4)


class ImageJitter(object):
    def __init__(self, transform_params):
        self.transforms = [
            (TRANSFORM_TYPES[k], transform_params[k]) for k in transform_params
        ]

    def __call__(self, img):
        out = img
        random_tensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (random_tensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")

        return out


class TransformLoader:
    def __init__(
        self,
        image_size,
        normalize_param=None,
        jitter_param=None,
    ):
        self.image_size = image_size
        self.normalize_param = normalize_param if normalize_param else NORMALIZE_DEFAULT
        self.jitter_param = jitter_param if jitter_param else JITTER_DEFAULT

    def parse_transform(
        self, transform_type
    ):  # Returns transformation method from its String name
        if (
            transform_type == "ImageJitter"
        ):  # Change Brightness, Constrast, Color and Sharpness randomly
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == "RandomResizedCrop":
            return method(self.image_size)
        elif transform_type == "CenterCrop":
            return method(self.image_size)
        elif transform_type == "Resize":
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == "Normalize":
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):  # Returns composed transformation
        if aug:
            transform_list = [
                "RandomResizedCrop",
                "ImageJitter",
                "RandomHorizontalFlip",
                "ToTensor",
                "Normalize",
            ]
        else:
            transform_list = [
                "Resize",
                "CenterCrop",
                "ToTensor",
                # "Normalize",
            ]

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform
