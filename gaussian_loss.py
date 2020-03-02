import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import numpy as np

def kl_div(mu_p, s2_p, mu_q=None, s2_q=None):
    """
    Computes KL Divergence of two diagonal multivariate Gaussian distributions p and q
    """
    if mu_q is None:
        mu_q = torch.zeros_like(mu_p)
        s2_q = torch.ones_like(mu_p)

    p = Normal(mu_p, torch.sqrt(s2_p))
    q = Normal(mu_q, torch.sqrt(s2_q))
    kl = kl_divergence(p, q).sum(dim=1)
    return kl


def gaussian_log_likelihood(x, mu, sigma2):
    """
    Computes log N(x|mu,sigma2)
    """
    term1 = torch.log(2 * np.pi * sigma2)
    term2 = (x - mu) ** 2 / sigma2
    log_lik = -0.5 * (term1 + term2).sum(dim=1)
    return log_lik


def mvrnorm(mu, s2):
    """
    Draw a random sample from N(mu, s2)
    """
    dist = Normal(mu, s2.sqrt())
    smp = dist.rsample()
    return smp


class GaussianLoss(nn.Module):
    """
    Gaussian loss function
    """

    def __init__(self, G: int, equal_variance: bool = False):
        super().__init__()
        self.G = G
        if not equal_variance:
            self.log_sigma = torch.Tensor([np.log(0.1)]) #torch.nn.Parameter(torch.randn(G), requires_grad=True)
        else:
            raise NotImplementedError

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    @property
    def sigma2(self):
        return torch.exp(2 * self.log_sigma)

    def forward(self, x, mu):
        loss = - gaussian_log_likelihood(x, mu, self.sigma2)
        #print(self.sigma2)
        return loss
