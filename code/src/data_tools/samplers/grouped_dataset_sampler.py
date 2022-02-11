import random

import numpy as np
import torch
from torch.utils.data import Sampler

from configs.evaluation_config import SUPPORT_QUERY_SHIFT


class GroupedDatasetSampler(Sampler):
    def __init__(self, dataset, n_way, n_source, n_target, n_episodes):
        self.n_way = n_way
        self.n_source = n_source
        self.n_target = n_target
        self.n_episodes = n_episodes

        self.meta_data = dataset.meta_data[["user", "class_name"]]
        self.class_list = list(self.meta_data.class_name.unique())
        self.domain_list = list(self.meta_data.user.unique())

        self.user_class_occurrences = (
            self.meta_data.groupby(["user", "class_name"])
            .size()
            .rename("n_images")
            .reset_index()
        )

        self.eligible_pairs_source = self.user_class_occurrences[
            ["user", "class_name"]
        ].loc[self.user_class_occurrences.n_images >= n_source]
        self.eligible_pairs_target = self.user_class_occurrences[
            ["user", "class_name"]
        ].loc[self.user_class_occurrences.n_images >= n_target]

        self.eligible_pairs_no_shift = self.user_class_occurrences[
            ["user", "class_name"]
        ].loc[self.user_class_occurrences.n_images >= n_source + n_target]
        domains_are_eligible_no_shift = self.eligible_pairs_no_shift.user.value_counts(
            sort=False
        ).gt(n_way)
        self.eligible_domains_no_shift = domains_are_eligible_no_shift.index[
            domains_are_eligible_no_shift
        ]

    def __len__(self):
        return self.n_episodes

    def _sample_domains_and_labels(self):
        for _ in range(20):
            source_domain = np.random.choice(
                self.eligible_pairs_source.user.unique(), 1
            )[0]
            target_domain = np.random.choice(
                self.eligible_pairs_target.user.unique(), 1
            )[0]
            if source_domain != target_domain:
                eligible_labels = set(
                    self.eligible_pairs_source.class_name.loc[
                        self.eligible_pairs_source.user == source_domain
                    ]
                ).intersection(
                    set(
                        self.eligible_pairs_target.class_name.loc[
                            self.eligible_pairs_target.user == target_domain
                        ]
                    )
                )
                if len(eligible_labels) >= self.n_way:
                    labels = random.sample(eligible_labels, self.n_way)
                    return source_domain, target_domain, labels
        raise TimeoutError(
            "Couldn't find a suitable task to sample (too many trials). Consider reducing task size."
        )

    def _sample_instances(self, label, domain, n_samples):
        eligible_indices = self.meta_data.index[
            (self.meta_data.user == domain) & (self.meta_data.class_name == label)
        ]
        return torch.tensor(
            eligible_indices
            if n_samples == -1
            else random.sample(list(eligible_indices), n_samples)
        )

    def _get_episode_items(self):
        if SUPPORT_QUERY_SHIFT:
            source_domain, target_domain, labels = self._sample_domains_and_labels()

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

        else:
            domain = np.random.choice(self.eligible_domains_no_shift, 1)[0]
            labels = np.random.choice(
                self.eligible_pairs_no_shift.class_name.loc[
                    self.eligible_pairs_no_shift.user == domain
                ],
                self.n_way,
                replace=False,
            )
            sampled_items = torch.stack(
                [
                    self._sample_instances(label, domain, self.n_source + self.n_target)
                    for label in labels
                ]
            )
            source_items = sampled_items[:, : self.n_source].flatten()
            target_items = sampled_items[:, self.n_source :].flatten()

        return torch.cat((source_items, target_items))

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self._get_episode_items()