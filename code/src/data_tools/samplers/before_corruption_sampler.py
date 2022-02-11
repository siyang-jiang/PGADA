import random

import torch
from torch.utils.data import Sampler

from sklearn.model_selection import train_test_split

from configs.evaluation_config import SUPPORT_QUERY_SHIFT


class BeforeCorruptionSampler(Sampler):
    """
    Sample images from a dataset which uses image perturbations (like CIFAR100-C or tieredImageNet-C), in the
    case where perturbations are applied online. For the case where images on disk are already corrupted, see
    AfterCorruptionSampler.
    """

    def __init__(self, dataset, n_way, n_source, n_target, n_episodes):
        self.n_domains = len(dataset.id_to_domain)
        self.n_total_images = len(dataset.images)
        self.n_way = n_way
        self.n_source = n_source
        self.n_target = n_target
        self.n_episodes = n_episodes

        self.items_per_label = {}

        for item, label in enumerate(dataset.labels):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self):
        return self.n_episodes

    def _split_source_target(self, labels):
        source_items_per_label = {}
        target_items_per_label = {}

        for label in labels:
            (
                source_items_per_label[label],
                target_items_per_label[label],
            ) = train_test_split(self.items_per_label[label], train_size=0.5)
        return source_items_per_label, target_items_per_label

    @staticmethod
    def _sample_instances(items, n_samples):
        return torch.tensor(
            items if n_samples == -1 else random.sample(items, n_samples)
        )

    def _get_episode_items(self):
        labels = random.sample(self.items_per_label.keys(), self.n_way)

        source_items_per_label, target_items_per_label = self._split_source_target(
            labels
        )

        if SUPPORT_QUERY_SHIFT:
            source_perturbation, target_perturbation = torch.randperm(self.n_domains)[
                :2
            ]
        else:
            source_perturbation = torch.randperm(self.n_domains)[:1]
            target_perturbation = source_perturbation

        source_items = (
            torch.cat(
                [
                    self._sample_instances(source_items_per_label[label], self.n_source)
                    for label in labels
                ]
            )
            * self.n_domains
            + source_perturbation
        )

        target_items = (
            torch.cat(
                [
                    self._sample_instances(target_items_per_label[label], self.n_target)
                    for label in labels
                ]
            )
            * self.n_domains
            + target_perturbation
        )

        return torch.cat((source_items, target_items))

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self._get_episode_items()
