import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, P, Q):
        p = F.softmax(P, dim=-1)
        kl = torch.sum(p * (F.log_softmax(P, dim=-1) - F.log_softmax(Q, dim=-1)))

        return torch.mean(kl)


class JSDivergence(nn.Module):
    def __init__(self):
        super(JSDivergence, self).__init__()
        self.kld = KLDivergence()

    def forward(self, P, Q):
        M = 0.5 * (P + Q)
        return 0.5 * (self.kld(P, M) + self.kld(Q, M))


class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        # print(input)
        # print(F.cross_entropy(input, target, reduction='mean'))
        # return F.cross_entropy(input, target, reduction='mean') * self.train_size + beta * kl
        # nll_loss = F.cross_entropy(input, target, reduction='mean') * self.train_size
        nll_loss = F.nll_loss(input, target, reduction='mean') * self.train_size
        kl_loss = beta * kl
        total_loss = nll_loss + kl_loss
        return total_loss, nll_loss, kl_loss 


class SVDD(nn.Module):
    def __init__(self, train_size):
        super(SVDD, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        # print(input)
        # print(F.cross_entropy(input, target, reduction='mean'))
        # return F.cross_entropy(input, target, reduction='mean') * self.train_size + beta * kl
        nll_loss = F.nll_loss(input, target, reduction='mean') * self.train_size
        kl_loss = beta * kl
        total_loss = nll_loss + kl_loss
        return total_loss, nll_loss, kl_loss


def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).mean()
    return kl


def dir_kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


class EdlLoss:
    def __init__(self, device):
        self.device = device

    def edl_loss(self, func, y, alpha, epoch_num, num_classes, annealing_step):
        y = y.to(self.device)
        alpha = alpha.to(self.device)
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * dir_kl_divergence(kl_alpha, num_classes, device=self.device)
        return A + kl_div

    def edl_digamma_loss(self, output, target, epoch_num, num_classes, annealing_step):
        evidence = F.relu(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.edl_loss(
                torch.digamma, target, alpha, epoch_num, num_classes, annealing_step
            )
        )
        return loss


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta