import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Normal, Gamma, MultivariateNormal
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
        # self.log_sigma = nn.Parameter(torch.zeros(self.d))
        # self.W = nn.Parameter(torch.zeros(self.d, self.K))
        # self.mu = nn.Parameter(torch.zeros(self.d))
        self.eps = 1e-6

        # ARD
        # Multiplies each column of W by the corresponding element of alpha (to nullify some of them)
        # self.log_alpha = nn.Parameter(torch.zeros(self.K))

        self.num_parameters = self.d * self.K + self.d + self.d + self.K # W, mu, log_sigma, log_alpha

        self.named_params = ["W", "mu", "log_sigma", "log_alpha"]
        self.dim_parameters = {"W": self.d*self.K, "mu": self.d, "log_sigma": self.d, "log_alpha": self.K}
        self.key_pos = ["log_sigma", "log_alpha"]

    def dist(self, W, mu, sigma, alpha):
        """Returns the distribution of the model"""
        # Multiply the column i of W by alpha_i
        W_multiplied = W.view(self.d, self.K) * alpha
        var_noise = sigma*torch.eye(self.d) + self.eps # Throws an error if the noise variance is 0
        mean = mu
        return MultivariateNormal(mean, torch.mm(W_multiplied, W_multiplied.t()) + var_noise)
    
    def rsample(self, n):
        """Samples n points from the distribution"""
        return self.dist().rsample([n])
    
    def theta_from_zeta(self, zeta):
        """Returns the model parameters from the tensor zeta"""
        W = zeta[:self.d*self.K]
        mu = zeta[self.d*self.K:self.d*self.K+self.d]
        sigma = torch.exp(zeta[self.d*self.K+self.d:self.d*self.K+2*self.d])
        alpha = torch.exp(zeta[self.d*self.K+2*self.d:])

        # concatenates the paremeters and returns them as one single tensor
        return torch.cat([W, mu, sigma, alpha])
    
    def grad_inv_T(self, zeta):
        """Returns the gradient of the inverse transformation"""
        with torch.no_grad():
            W = torch.ones(self.d*self.K)
            mu = torch.ones(self.d)
            sigma = torch.exp(zeta[self.d*self.K+self.d:self.d*self.K+2*self.d])
            alpha = torch.exp(zeta[self.d*self.K+2*self.d:])
            
            return torch.cat([W, mu, sigma, alpha])
        
    def log_det(self, zeta):
        """Returns the log determinant of the jacobian of the inverse transformation"""
        return torch.sum(zeta[self.d*self.K+self.d:self.d*self.K+2*self.d]) + torch.sum(zeta[self.d*self.K+2*self.d:])
    
    def log_prob(self, x, theta, full_data_size):
        """Returns the log probability of x under the distribution of the model"""
        # gets the model parameters from the tensor params
        W = theta[:self.d*self.K].view(self.d, self.K)
        mu = theta[self.d*self.K:self.d*self.K+self.d]
        sigma = theta[self.d*self.K+self.d:self.d*self.K+2*self.d]
        alpha = theta[self.d*self.K+2*self.d:]

        # sums the log probabilities of each point
        sum = 0.
        for i in range(x.shape[0]):
            sum += self.dist(W, mu, sigma, alpha).log_prob(x[i]).sum()
        return sum*full_data_size/x.shape[0]
        