import torch
from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.methods.utils import euclidean_dist
from src.utils import set_device
from src.modules.hyper_networks import MultiLayerPerceptron, RelationNet


class TransPropNet(AbstractMetaLearner):
    def __init__(
        self,
        model_func,
        transportation=None,
        training_stats=None,
        alpha=0.99,
        eps=1e-8,
        k=20,
    ):
        super(TransPropNet, self).__init__(
            model_func, transportation=transportation, training_stats=training_stats
        )

        # Hypernetwork that fits scaling factor in similarity
        self.length_scale = RelationNet(self.feature.final_feat_dim)

        # Hyper-parameters used in the paper.
        self.alpha = alpha
        self.eps = eps
        self.k = k

    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """

        z_support, z_query = self.extract_features(support_images, query_images)

        # If a transportation method in the feature space has been defined, use it
        if self.transportation_module:
            z_support, z_query = self.transportation_module(z_support, z_query)

        similarity = self.get_similarity(z_support, z_query)

        # Normalization of the laplacian
        normalize = (1.0 / (similarity + self.eps).sum(dim=0)).diag().sqrt()

        laplacian = torch.matmul(normalize, torch.matmul(similarity, normalize))

        scores = self.propagate(laplacian, support_labels)

        return scores[support_images.size(0) :]

    def get_similarity(self, z_support, z_query):
        """
        Compute the similarity matrix sample to sample for label propagation.
        Note that support and query are merged.
        See eq (2) of LEARNING TO PROPAGATE LABELS: TRANSDUCTIVE PROPAGATION NETWORK FOR FEW-SHOT LEARNING
        Args:
            z_support (torch.Tensor): shape (n_support, features_dim)
            z_query (torch.Tensor): shape (n_query, features_dim)
        Returns:
            torch.Tensor: shape(n_support + n_query, n_support + n_query), similarity matrix between samples.
        """
        # scaling with forward of self.length_scale
        z_support = z_support / (self.length_scale(z_support) + self.eps)
        z_query = z_query / (self.length_scale(z_query) + self.eps)

        z = torch.cat([z_support, z_query], dim=0)

        similarity = torch.exp(-0.5 * euclidean_dist(z, z) / z.shape[1])

        if similarity.shape[1] > self.k:
            # Keep only top k values in the similarity matrix, set 0. otherwise.
            _, indices = similarity.topk(self.k, dim=1)
            similarity = similarity - similarity.scatter(1, indices, 0)

        return similarity

    def propagate(self, laplacian, support_labels):
        """
        Compute label propagation.
        See eq (4) of LEARNING TO PROPAGATE LABELS: TRANSDUCTIVE PROPAGATION NETWORK FOR FEW-SHOT LEARNING
        Args:
            laplacian (torch.Tensor): shape (n_support + n_query, n_support + n_query)
            support_labels (torch.Tensor): artificial support set labels in range (0, n_way)
        Returns:
            torch.Tensor: shape(n_support + n_query, n_support + n_query), similarity matrix between samples.
        """

        # compute labels as one_hot
        n_way = len(torch.unique(support_labels))
        n_support_query = laplacian.size(0)
        n_support = support_labels.size(0)
        n_query = n_support_query - n_support

        ## compute support labels as one hot
        one_hot_labels = set_device(torch.zeros(n_support, n_way))

        one_hot_labels[torch.arange(n_support), support_labels] = 1.0

        ## sample to predict has 0 everywhere
        one_hot_labels = torch.cat(
            [one_hot_labels, set_device(torch.zeros(n_query, n_way))]
        )

        # compute label propagation
        propagation = (
            set_device(torch.eye(laplacian.size(0))) - self.alpha * laplacian + self.eps
        ).inverse()

        scores = torch.matmul(propagation, one_hot_labels)

        return scores
