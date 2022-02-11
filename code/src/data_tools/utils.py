import cv2
from functools import partial
import re

import numpy as np
import torch

from src.data_tools.perturbations import PERTURBATIONS


def episodic_collate_fn(input_data):
    """
    Collate function to be used as argument for the collate_fn parameter of episodic data loaders.
    Args:
        input_data (list[tuple(Tensor, int, int)]): each element is a tuple containing:
            - an image as a torch Tensor
            - the label of this image
            - the group of this image

    Returns:
        tuple(Tensor, Tensor, Tensor, Tensor, list[int], int, int): respectively:
            - source images,
            - their labels,
            - target images,
            - their labels,
            - the dataset class ids of the class sampled in the episode
            - source domain
            - target domain
    """
    source = input_data[0][2]
    target = input_data[-1][2]
    assert all(item[2] == source or item[2] == target for item in input_data)

    true_class_ids = list(set([x[1] for x in input_data]))

    images_source = torch.cat([x[0].unsqueeze(0) for x in input_data if x[2] == source])
    labels_source = torch.tensor(
        [true_class_ids.index(x[1]) for x in input_data if x[2] == source]
    )

    images_target = torch.cat([x[0].unsqueeze(0) for x in input_data if x[2] == target])
    labels_target = torch.tensor(
        [true_class_ids.index(x[1]) for x in input_data if x[2] == target]
    )

    return (
        images_source,
        labels_source,
        images_target,
        labels_target,
        true_class_ids,
        int(source),
        int(target),
    )


def get_perturbations(perturbation_specs, perturbation_params, image_size):
    """
    Retrieve perturbation function from dataset specs.
    Args:
        perturbation_specs (dict): keys must belong to the keys of PERTURBATIONS,
            values are lists of integers
        perturbation_params (dict): keys are exactly the keys of PERTURBATIONS,
            values are lists where each element is a possible additional parameter for a perturbation
        image_size (int): expected image size

    Returns:
        tuple(list, dict): respectively:
            - list of partial functions of perturbations, where the severity is already set
            - dictionary associating any integer id to the name of the corresponding perturbation
                in the previous list

    """
    perturbations = []
    id_to_domain_list = []
    for perturbation_name, severities in perturbation_specs.items():
        for severity in severities:
            perturbations.append(
                partial(
                    PERTURBATIONS[perturbation_name],
                    severity_params=perturbation_params[perturbation_name][
                        severity - 1
                    ],
                    image_size=image_size,
                )
            )
            id_to_domain_list.append(f"{clean_name(perturbation_name)}_{severity}")
    id_to_domain = dict(enumerate(id_to_domain_list))

    return perturbations, id_to_domain


def clean_name(name, banned_characters="[^A-Za-z0-9]+", fill_item="_"):
    return re.sub(banned_characters, fill_item, name)


def load_image_as_array(filename, image_size):
    return np.asarray(
        cv2.resize(cv2.imread(filename)[..., ::-1], (image_size, image_size))
    )
