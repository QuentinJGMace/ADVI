import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Normal, Gamma, MultivariateNormal
import numpy as np

class PPCA_model(nn.Module):
    """
    Class for a PPCA model
    """

    def __init__(self, D, K):
        """
        D: int, data dimension
        K: int, latent dimension
        """
        super(PPCA_model, self).__init__()
        self.K = K # latent dimension
        self.d = D # data dimension

        self.eps = 1e-6

        # Variables used for ADVI to know the dimensionality of the optimization problem
        self.num_parameters = self.d * self.K + self.d + self.d # W, mu, log_sigma

        self.named_params = ["W", "mu", "log_sigma"]
        self.dim_parameters = {"W": self.d*self.K, "mu": self.d, "log_sigma": self.d}
        # Key_pos is now ununsed but it keeps the model variables that are supposed to be positive
        self.key_pos = ["log_sigma"]

    def dist(self, W, mu, sigma):
        """Returns the distribution of the model"""
        W_multiplied = W.view(self.d, self.K)
        if not (sigma >0.).all():
            breakpoint()
        var_noise = sigma*torch.eye(self.d) + self.eps # Throws an error if the noise variance is 0
        mean = mu
        # Try / Except to understand errors
        try:
            return MultivariateNormal(mean, torch.mm(W_multiplied, W_multiplied.t()) + var_noise)
        except:
            print("Ws", W_multiplied, W_multiplied.t())
            print("sig", sigma)
            print("var", var_noise)
            1/0
    
    def rsample(self, n, W, mu, sigma):
        """Samples n points from the distribution"""
        return self.dist(W, mu, sigma).rsample([n])
    
    def theta_from_zeta(self, zeta:torch.Tensor):
        """Returns the model parameters from the tensor zeta (applies T^-1)"""
        W = zeta[:self.d*self.K]
        mu = zeta[self.d*self.K:self.d*self.K+self.d]
        sigma = torch.exp(zeta[self.d*self.K+self.d:self.d*self.K+2*self.d])

        assert (sigma > 0.).all(), "Sigma should be positive"

        # concatenates the paremeters and returns them as one single tensor
        return torch.cat([W, mu, sigma])
    
    def grad_inv_T(self, zeta:torch.Tensor):
        """
        DEPRECATED
        Returns the gradient of the inverse transformation"""
        with torch.no_grad():
            W = torch.ones(self.d*self.K)
            mu = torch.ones(self.d)
            sigma = torch.exp(zeta[self.d*self.K+self.d:self.d*self.K+2*self.d])
            
            return torch.cat([W, mu, sigma])
        
    def log_det(self, zeta:torch.Tensor):
        """Returns the log determinant of the jacobian of the inverse transformation"""
        return torch.sum(zeta[self.d*self.K+self.d:self.d*self.K+2*self.d])
    
    def log_prob(self, x:torch.Tensor, theta:torch.Tensor, full_data_size:int):
        """Returns the log probability of x under the distribution of the model
        
        Args:
            x: torch.Tensor, data
            theta: torch.Tensor, model parameters
            full_data_size: int, size of the full data set (used for batch learning)"""
        # gets the model parameters from the tensor params
        W = theta[:self.d*self.K].view(self.d, self.K)
        mu = theta[self.d*self.K:self.d*self.K+self.d]
        sigma = theta[self.d*self.K+self.d:self.d*self.K+2*self.d]

        assert(sigma > 0.).all() # Debugging

        # Sums (and scales for batch learning) the log probabilities of each point
        sum = 0.
        for i in range(x.shape[0]):
            sum += self.dist(W, mu, sigma).log_prob(x[i]).sum()
        return sum*full_data_size/x.shape[0]
        

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

        self.eps = 1e-6

        self.num_parameters = self.d * self.K + self.d + self.d + self.K # W, mu, log_sigma, log_alpha

        self.named_params = ["W", "mu", "log_sigma", "log_alpha"]
        self.dim_parameters = {"W": self.d*self.K, "mu": self.d, "log_sigma": self.d, "log_alpha": self.K}
        self.key_pos = ["log_sigma", "log_alpha"]

    def dist(self, W:torch.Tensor, mu:torch.Tensor, sigma:torch.Tensor, alpha:torch.Tensor):
        """Returns the distribution of the model"""
        # Multiply the column i of W by alpha_i (ARD)
        W_multiplied = W.view(self.d, self.K) * alpha
        var_noise = sigma*torch.eye(self.d) + self.eps # Throws an error if the noise variance is 0
        mean = mu
        try:
            return MultivariateNormal(mean, torch.mm(W_multiplied, W_multiplied.t()) + var_noise)
        except:
            print("Ws", W_multiplied, W_multiplied.t())
            print("sig", sigma)
            print("alpha", alpha)
            print("var", var_noise)
            raise Exception("Error in the distribution")
    
    def rsample(self, n:int, W:torch.Tensor, mu:torch.Tensor, sigma:torch.Tensor, alpha:torch.Tensor):
        """Samples n points from the distribution"""
        return self.dist(W, mu, sigma, alpha).rsample([n])
    
    def theta_from_zeta(self, zeta:torch.Tensor):
        """Returns the model parameters from the tensor zeta"""
        W = zeta[:self.d*self.K]
        mu = zeta[self.d*self.K:self.d*self.K+self.d]
        sigma = torch.exp(zeta[self.d*self.K+self.d:self.d*self.K+2*self.d])
        alpha = torch.exp(zeta[self.d*self.K+2*self.d:])

        assert (sigma > 0.).all(), "Sigma should be positive"
        assert (alpha >0.).all(), "Alpha should be positive"

        # concatenates the paremeters and returns them as one single tensor
        return torch.cat([W, mu, sigma, alpha])
    
    def grad_inv_T(self, zeta:torch.Tensor):
        """Returns the gradient of the inverse transformation"""
        with torch.no_grad():
            W = torch.ones(self.d*self.K)
            mu = torch.ones(self.d)
            sigma = torch.exp(zeta[self.d*self.K+self.d:self.d*self.K+2*self.d])
            alpha = torch.exp(zeta[self.d*self.K+2*self.d:])
            
            return torch.cat([W, mu, sigma, alpha])
        
    def log_det(self, zeta:torch.Tensor):
        """Returns the log determinant of the jacobian of the inverse transformation"""
        return torch.sum(zeta[self.d*self.K+self.d:self.d*self.K+2*self.d]) + torch.sum(zeta[self.d*self.K+2*self.d:])
    
    def log_prob(self, x:torch.Tensor, theta:torch.Tensor, full_data_size:int):
        """Returns the log probability of x under the distribution of the model"""
        # gets the model parameters from the tensor params
        W = theta[:self.d*self.K].view(self.d, self.K)
        mu = theta[self.d*self.K:self.d*self.K+self.d]
        sigma = theta[self.d*self.K+self.d:self.d*self.K+2*self.d]
        alpha = theta[self.d*self.K+2*self.d:]

        assert(sigma > 0.).all()
        assert (alpha > 0.).all()

        # sums the log probabilities of each point
        sum = 0.
        for i in range(x.shape[0]):
            sum += self.dist(W, mu, sigma, alpha).log_prob(x[i]).sum()
        return sum*full_data_size/x.shape[0]
        