"""
This code is extracted from https://github.com/yhu01/PT-MAP/blob/master/test_standard.py
We only added minimal fixes to make it work in our code base.
We can't promise anything about the quality of the code in this file.
"""


import torch
import math

from torch import nn
from tqdm import tqdm

from configs import evaluation_config
from src.methods.abstract_meta_learner import AbstractMetaLearner


# ========================================
#      loading datas


def centerDatas(datas, n_lsamples):
    support_means = datas[:, :n_lsamples].mean(1, keepdim=True)
    query_means = datas[:, n_lsamples:].mean(1, keepdim=True)
    support_norm = torch.norm(datas[:, :n_lsamples], 2, 2)[:, :, None]
    query_norm = torch.norm(datas[:, n_lsamples:], 2, 2)[:, :, None]

    datas_out = torch.zeros_like(datas)
    datas_out[:, :n_lsamples] = (datas[:, :n_lsamples] - support_means) / support_norm
    datas_out[:, n_lsamples:] = (datas[:, n_lsamples:] - query_means) / query_norm

    return datas_out


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.qr(datas.permute(0, 2, 1)).R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


# ---------  GaussianModel
class GaussianModel:
    def __init__(
        self,
        n_ways,
        lam,
        ndatas,
        n_runs,
        n_shot,
        n_queries,
        n_nfeat,
        n_lsamples,
        n_usamples,
    ):
        self.n_ways = n_ways
        self.mus = None  # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        self.n_runs = n_runs
        self.n_queries = n_queries
        self.n_lsamples = n_lsamples
        self.n_usamples = n_usamples
        self.mus = ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[
            :,
            :n_shot,
        ].mean(1)

    def cuda(self):
        # Inplace
        self.mus = self.mus.cuda()

    def updateFromEstimate(self, estimate, alpha):

        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):

        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(-self.lam * M)
        P = P / P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)

        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P = P * (r / u).view((n_runs, -1, 1))
            P = P * (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)

    def getProbas(self, ndatas, labels):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2) - self.mus.unsqueeze(1)).norm(dim=3).pow(2)

        p_xj = torch.zeros_like(dist)
        r = torch.ones(self.n_runs, self.n_usamples)
        c = torch.ones(self.n_runs, self.n_ways) * self.n_queries

        p_xj_test, _ = self.compute_optimal_transport(
            dist[:, self.n_lsamples :], r, c, epsilon=1e-6
        )
        p_xj[:, self.n_lsamples :] = p_xj_test

        p_xj[:, : self.n_lsamples] = p_xj[:, : self.n_lsamples].scatter(
            2, labels[:, : self.n_lsamples].unsqueeze(2), 1
        )

        return p_xj

    def estimateFromMask(self, mask, ndatas):

        emus = mask.permute(0, 2, 1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus


class MAP(AbstractMetaLearner):
    def __init__(self, model_func, transportation=None, power_transform=True):
        super().__init__(model_func)
        self.loss_fn = nn.NLLLoss()
        self.power_transform = power_transform

    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """
        n_shot = evaluation_config.N_SOURCE_EVAL
        n_ways = evaluation_config.N_WAY_EVAL
        n_queries = evaluation_config.N_TARGET_EVAL
        n_runs = 1
        n_lsamples = n_ways * n_shot
        n_usamples = n_ways * n_queries
        n_samples = n_lsamples + n_usamples
        self.n_lsamples = n_lsamples
        self.n_runs = n_runs

        z_support, z_query = self.extract_features(support_images, query_images)
        label_mapping = support_labels.view(n_ways, n_shot).permute(1, 0).sort()[1][0]

        support_mapping = torch.cat([label_mapping * n_shot + i for i in range(n_shot)])
        query_mapping = torch.cat(
            [label_mapping * n_queries + i for i in range(n_queries)]
        )

        ndatas = torch.cat(
            (z_support[support_mapping], z_query[query_mapping]), dim=0
        ).unsqueeze(0)
        labels = (
            torch.arange(n_ways)
            .view(1, 1, n_ways)
            .expand(n_runs, n_shot + n_queries, 5)
            .clone()
            .view(n_runs, n_samples)
        )

        if self.power_transform:
            # Power transform
            beta = 0.5
            ndatas[:,] = torch.pow(
                ndatas[
                    :,
                ]
                + 1e-6,
                beta,
            )

        ndatas = QRreduction(ndatas)  # Now ndatas has shape (1, n_samples, n_samples)
        n_nfeat = ndatas.size(2)

        ndatas = scaleEachUnitaryDatas(ndatas)

        # trans-mean-sub

        ndatas = centerDatas(ndatas, n_lsamples)

        print("size of the datas...", ndatas.size())

        # switch to cuda
        ndatas = ndatas.cuda()
        labels = labels.cuda()

        # MAP
        lam = 10
        model = GaussianModel(
            n_ways,
            lam,
            ndatas,
            n_runs,
            n_shot,
            n_queries,
            n_nfeat,
            n_lsamples,
            n_usamples,
        )

        self.alpha = 0.2

        self.ndatas = ndatas
        self.labels = labels

        probas = self.loop(model, n_epochs=20)

        # TODO remettre les labels dans le sens originel

        return probas.squeeze(0)[n_lsamples:][query_mapping.sort()[1]]

    def getAccuracy(self, probas, labels):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:, self.n_lsamples :].mean(1)

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(self.n_runs)
        return m, pm

    def performEpoch(self, model, epochInfo=None):
        p_xj = model.getProbas(self.ndatas, self.labels)
        self.probas = p_xj

        m_estimates = model.estimateFromMask(self.probas, self.ndatas)

        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

    def loop(self, model, n_epochs=20):
        self.probas = model.getProbas(self.ndatas, self.labels)

        for epoch in tqdm(range(1, n_epochs + 1)):
            self.performEpoch(model, epochInfo=(epoch, n_epochs))

        # get final accuracy and return it
        op_xj = model.getProbas(self.ndatas, self.labels)
        return op_xj
