import random

import torch
from torch.utils.data import Sampler


class AfterCorruptionSampler(Sampler):
    def __init__(self, dataset, n_way, n_source, n_target, n_episodes):
        self.data = dataset.images_df[["class_id", "domain_id"]]
        self.class_list = list(self.data.class_id.unique())
        self.domain_list = list(self.data.domain_id.unique())
        self.n_way = n_way
        self.n_source = n_source
        self.n_target = n_target
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def _sample_instances(self, label, domain, n_samples):
        eligible_indices = self.data.index[
            (self.data.domain_id == domain) & (self.data.class_id == label)
        ]
        return torch.tensor(
            eligible_indices
            if n_samples == -1
            else random.sample(list(eligible_indices), n_samples)
        )

    def _get_episode_items(self):
        labels = random.sample(self.class_list, self.n_way)
        source_domain, target_domain = random.sample(self.domain_list, 2)

        source_items = torch.cat(
            [
                self._sample_instances(label, source_domain, self.n_source)
                for label in labels
            ]
        )

        target_items = torch.cat(
            [
                self._sample_instances(label, target_domain, self.n_target)
                for label in labels
            ]
        )

        return torch.cat((source_items, target_items))

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self._get_episode_items()
