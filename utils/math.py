import torch
from torch.distributions import Normal

def gaussian_kl(mean1, std1, mean2, std2):
    """
    Calculate KL-divergence between two Gaussian distributions N(mu1, sigma1) and N(mu2, sigma2)
    """
    normal1 = Normal(mean1, std1)
    normal2 = Normal(mean2, std2)
    return torch.distributions.kl.kl_divergence(normal1,normal2).sum(-1, keepdim=True)
