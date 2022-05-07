"""
Implement Vinyals et al. (Matching networks for One-Shot Learning)
"""

import torch
import torch.nn as nn

from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.methods import utils
from src.modules.hyper_networks import FullyContextualEmbedding
from src.utils import set_device


class MatchingNet(AbstractMetaLearner):
    def __init__(self, model_func, transportation=None, training_stats=None):
        super(MatchingNet, self).__init__(
            model_func, transportation=transportation, training_stats=training_stats
        )

        self.loss_fn = nn.NLLLoss()

        self.FCE = FullyContextualEmbedding(self.feature.final_feat_dim)
        self.support_features_encoder = set_device(
            nn.LSTM(
                self.feature.final_feat_dim,
                self.feature.final_feat_dim,
                1,
                batch_first=True,
                bidirectional=True,
            )
        )

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode_support_features(
        self,
        support_features: torch.Tensor,
    ):
        encoded_support_features = self.support_features_encoder(
            support_features.unsqueeze(0)
        )[0].squeeze(0)
        encoded_support_features = (
            support_features
            + encoded_support_features[:, : support_features.size(1)]
            + encoded_support_features[:, support_features.size(1) :]
        )
        normalized_encoded_support_features = encoded_support_features.div(
            torch.norm(encoded_support_features, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(encoded_support_features)
            + 0.00001
        )
        return encoded_support_features, normalized_encoded_support_features

    def get_logprobs(
        self,
        query_features,
        encoded_support_features,
        normalized_encoded_support_features,
        one_hot_support_labels,
    ):

        contextualized_query_features = self.FCE(
            query_features, encoded_support_features
        )
        scores = self.softmax(
            self.relu(
                contextualized_query_features.div(
                    torch.norm(contextualized_query_features, p=2, dim=1)
                    .unsqueeze(1)
                    .expand_as(contextualized_query_features)
                    + 0.00001
                ).mm(normalized_encoded_support_features.transpose(0, 1))
            )
            * 100
        )
        logprobs = (scores.mm(one_hot_support_labels) + 1e-6).log()
        return logprobs

    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """
        z_support, z_query = self.extract_features(support_images, query_images)
        # If a transportation method in the feature space has been defined, use it
        if self.transportation_module:
            z_support, z_query = self.transportation_module(z_support, z_query)

        (
            encoded_support_features,
            encoded_support_features,
        ) = self.encode_support_features(z_support)

        support_labels_one_hot = utils.one_hot(support_labels)

        logprobs = self.get_logprobs(
            z_query,
            encoded_support_features,
            encoded_support_features,
            support_labels_one_hot,
        )
        return logprobs
