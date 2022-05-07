import torch
import torch.nn as nn
import numpy as np

from src.utils import set_device


def confidence_interval(standard_deviation, n_samples):
    """
    Computes statistical 95% confidence interval of the results from standard deviation and number of samples
    Args:
        standard_deviation (float): standard deviation of the results
        n_samples (int): number of samples

    Returns:
        float: confidence interval
    """
    return 1.96 * standard_deviation / np.sqrt(n_samples)


def one_hot(labels):
    """

    Args:
        labels (torch.Tensor): 1-dimensional tensor of integers

    Returns:
        torch.Tensor: 2-dimensional tensor of shape[len(labels), max(labels)] corresponding to the one-hot
            form of the input tensor
    """
    num_class = torch.max(labels) + 1
    return set_device(
        torch.zeros((len(labels), num_class)).scatter_(1, labels.unsqueeze(1), 1)
    )


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def softplus(x):
    return torch.log(1 + x.exp())


def entropy(logits):
    """
    Compute entropy of prediction.
    WARNING: takes logit as input, not probability.
    Args:
        logits (torch.Tensor): shape (, n_way)
    Returns:
        torch.Tensor: shape(), Mean entropy.
    """
    p = nn.Softmax(dim=1)(logits)
    entropy = -(p * (p + 1e-6).log()).sum(dim=1)
    return entropy.mean()
