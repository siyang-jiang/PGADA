import torch
from sklearn.linear_model import RidgeClassifier

from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.utils import set_device


class UnsupDomAdapOT(AbstractMetaLearner):
    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """

        support_query_size = len(support_images)
        n_chunks = support_query_size // 32 + 1

        support_chunk = []
        query_chunk = []

        for support, query in zip(
            support_images.chunk(n_chunks), query_images.chunk(n_chunks)
        ):

            support_features, query_features = (
                features.detach().cpu()
                for features in self.extract_features(
                    set_device(support), set_device(query)
                )
            )

            support_chunk.append(support_features.detach().cpu())
            query_chunk.append(query_features.detach().cpu())

        z_support = torch.cat(support_chunk, dim=0)

        del support_chunk

        z_query = torch.cat(query_chunk, dim=0)

        del query_chunk

        # If a transportation method in the feature space has been defined, use it
        if self.transportation_module:
            z_support, z_query = (
                z.cpu()
                for z in self.transportation_module(
                    set_device(z_support), set_device(z_query)
                )
            )

        z_support = z_support.numpy()
        z_query = z_query.numpy()
        support_labels = support_labels.cpu().numpy()

        linear_classifier = RidgeClassifier(alpha=0.1)
        linear_classifier.fit(z_support, support_labels)

        scores = torch.tensor(linear_classifier.decision_function(z_query))

        scores = set_device(scores)
        return scores

    def train_loop(self, epoch, train_loader, optimizer):
        raise NotImplementedError(
            "UDA-OT from Courty et al. does not support episodic training"
        )
