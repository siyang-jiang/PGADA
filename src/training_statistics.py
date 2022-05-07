import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch


class TrainingStatistics:
    def __init__(self, dataset):
        """

        Args:
            dataset (SetDataset): a set dataset on which we compute the statistics of meta-training
        """

        n_classes = len(dataset)
        self.confusion_matrix = pd.DataFrame(
            np.zeros((n_classes, n_classes)),
            index=dataset.label_list,
            columns=dataset.label_list,
        )
        self.label_count = pd.DataFrame(index=dataset.label_list)

    def update(self, prediction_scores, true_class_ids, epoch):
        """
        Update the training statistics based on the classifications from one episode.
        Args:
            prediction_scores (torch.Tensor): shape(n_query*n_way, n_way), classification prediction for each query data
            true_class_ids (list): value of index i is the true class id corresponding to artificial label i
            epoch (int): current epoch

        """
        self.update_confusion(prediction_scores, true_class_ids)
        self.update_label_count(true_class_ids, epoch)

    def update_confusion(self, prediction_scores, true_labels):
        """
        Update the confusion matrix based on the classifications from one episode.
        Args:
            prediction_scores (torch.Tensor): shape(n_query*n_way, n_way), classification prediction for each query data
            true_labels (list[int]): list of true labels composing the episode

        """
        n_way = len(true_labels)
        n_query = prediction_scores.size(0) // n_way
        assert isinstance(n_query, int)

        classification_gt = torch.from_numpy(np.repeat(range(n_way), n_query))
        classification = torch.argmax(prediction_scores.cpu(), dim=1)
        local_confusion = confusion_matrix(classification_gt, classification)

        for (local_label1, true_label1) in enumerate(true_labels):
            for (local_label2, true_label2) in enumerate(true_labels):
                self.confusion_matrix.at[true_label1, true_label2] = (
                    self.confusion_matrix.at[true_label1, true_label2]
                    + local_confusion[local_label1, local_label2]
                )

    def update_label_count(self, true_labels, epoch):
        """
        Updates the label_count matrix, i.e. the number of times each label was sampled in an episode during each epoch
        Args:
            true_labels (list[int]): list of true labels composing the episode
            epoch (int): current epoch
        """
        if epoch not in self.label_count.columns:
            self.label_count[epoch] = 0

        self.label_count.loc[true_labels, epoch] += 1


def compute_biconfusion_matrix(confusion_matrix):
    """
    The biconfusion matrix is typically used to measure the hardness of the discrimination task between two classes.
        Element (i,j) corresponds to the number of missclassifications between classes i and j, regardless of the
        direction (i to j or j to i).
    Args:
        confusion_matrix(pd.DataFrame): a 2-dimentional square matrix

    Returns:
        pd.DataFrame: a 2-dimentional symmetric square matrix of the same shape as confusion_matrix.
    """
    biconfusion_matrix = confusion_matrix.add(confusion_matrix.T)
    biconfusion_matrix.update(np.fill_diagonal(biconfusion_matrix.values, 0))

    return biconfusion_matrix
