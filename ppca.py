import torch
import torch.nn as nn
from torch.distributions import Normal, Gamma
import numpy as np

class PPCA_with_ARD_model(nn.Module):
    """
    Class for a PPCA model with ARD (Automatic Relevance Determination) for the latent dimensions
    """

    def __init__(self, D, K):
        """
        D: int, data dimension
        K: int, latent dimension
        """
        super(PPCA_with_ARD_model, self).__init__()
        self.K = K # latent dimension
        self.d = D # data dimension

        # Simple PPCA without ARD but with unconstrained noise variance
        self.log_sigma = nn.Parameter(torch.zeros(self.d))
        self.W = nn.Parameter(torch.zeros(self.d, self.K))
        self.mu = nn.Parameter(torch.zeros(self.d))

        # ARD
        # Multiplies each column of W by the corresponding element of alpha (to nullify some of them)
        self.log_alpha = nn.Parameter(torch.zeros(self.K))

    def dist(self):
        """Returns the distribution of the model"""
        # Multiply the column i of W by alpha_i
        W_multiplied = self.W * torch.exp(self.log_alpha)
        var_noise = self.log_sigma.exp()*torch.eye(self.d)
        mean = self.mu
        return Normal(mean, torch.mm(W_multiplied, W_multiplied.t()) + var_noise)
    
    def rsample(self, n):
        """Samples n points from the distribution"""
        return self.dist().rsample([n])
    
    def log_prob(self, x):
        """Returns the log probability of x under the distribution of the model"""
        return self.dist().log_prob(x)