import torch
from torch.autograd import Variable
from torch import nn

from src.methods.utils import softplus
from src.utils import set_device


class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_features):
        super(MultiLayerPerceptron, self).__init__()

        self.layer_1 = nn.Linear(num_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.layer_2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.layer_3 = nn.Linear(1024, 1)

        relu = nn.ReLU()

        self.mlp = nn.Sequential(
            self.layer_1, self.bn1, relu, self.layer_2, self.bn2, relu, self.layer_3
        )

    def forward(self, x):
        return softplus(self.mlp(x))


class RelationNet(nn.Module):
    """
    RelationNet for Transductive Propagation Net.
    Used for scaling the similarity.
    Uses a ConvNet architecture.
    """

    def __init__(self, num_features):
        super(RelationNet, self).__init__()

        self.lin_1 = nn.Linear(num_features, 64)
        self.lin_2 = nn.Linear(64, 8)
        self.lin_3 = nn.Linear(8, 1)

        # We enforce to start from sigma=0.1 to prevent learning failure. MANDATORY!
        self.lin_1.weight.data.fill_(0.0)
        self.lin_1.bias.data.fill_(1.0)
        self.lin_2.weight.data.fill_(0.0)
        self.lin_2.bias.data.fill_(1.0)
        self.lin_3.weight.data.fill_(0.0)
        self.lin_3.bias.data.fill_(1.0)

        relu = nn.ReLU()

        self.net = nn.Sequential(self.lin_1, relu, self.lin_2, relu, self.lin_3)

    def forward(self, x):
        sigma = self.net(x)
        return sigma


class FullyContextualEmbedding(nn.Module):
    """
    See Vinyals et al. (Matching networks for One-Shot Learning)
    """

    def __init__(self, num_features):
        super(FullyContextualEmbedding, self).__init__()
        self.lstm_cell = set_device(nn.LSTMCell(num_features * 2, num_features))
        self.softmax = nn.Softmax()
        self.c_0 = set_device(Variable(torch.zeros(1, num_features)))

    def forward(self, query_features, encoded_support_features):
        h = query_features
        c = self.c_0.expand_as(query_features)
        K = encoded_support_features.size(0)  # Tuna to be comfirmed
        for k in range(K):
            logit_a = h.mm(encoded_support_features.T)
            a = self.softmax(logit_a)
            r = a.mm(encoded_support_features)
            x = torch.cat((query_features, r), 1)

            h, c = self.lstm_cell(x, (h, c))
            h = h + query_features

        return h
