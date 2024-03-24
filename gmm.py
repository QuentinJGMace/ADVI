import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Normal, LogNormal, Dirichlet
import numpy as np
import torch
from torch.distributions.distribution import Distribution
from torch.distributions import Categorical
from torch.distributions import constraints
from typing import Dict
from torch.distributions.utils import _standard_normal
from pdb import set_trace

class GMM(nn.Module):
    """
    Class for a GMM
    """

    def __init__(self, D, K, alpha_0 = 1000):
        """
        D: int, data dimension
        K: int, number of gaussians
        """
        super(GMM, self).__init__()
        self.K = K # latent dimension
        self.d = D # data dimension
        self.alpha_0 = alpha_0 # Dirichlet prior for the weights

        self.eps = 1e-6

        # Variables used for ADVI to know the dimensionality of the optimization problem
        self.num_parameters = self.K + self.K * self.d + self.K * self.d # p, mu, log_sigma

        self.named_params = ["p", "mu", "log_sigma"]
        self.dim_parameters = {"p": self.K, "mu": self.d * self.K, "log_sigma": self.d * self.K}
        # Key_pos is now ununsed but it keeps the model variables that are supposed to be positive
        self.key_pos = ["log_sigma"]
    
    def rsample(self, n, p, mu, sigma):
        """Samples n points from the distribution"""
        shape = n
        eps = _standard_normal(shape*self.d, dtype=torch.float32, device="cpu").view(shape, self.d)
        
        # Calculate the mixture component indices based on the weights
        mixture_indices = Categorical(p).sample((shape,))
        
        # Select the means corresponding to the mixture component indices
        selected_means = mu[mixture_indices]
        selected_variances = sigma[mixture_indices]
        
        # Generate random samples by adding noise to the selected means
        samples = selected_means + eps * selected_variances
        
        return samples
    
    def theta_from_zeta(self, zeta:torch.Tensor):
        """Returns the model parameters from the tensor zeta (applies T^-1)"""
        zeta_p = zeta[:self.K]
        exp_sum = zeta_p.exp().sum() 
        p = zeta_p.exp() / exp_sum
        mu = zeta[self.K:self.K+self.K*self.d]
        sigma = torch.exp(zeta[self.K+self.K*self.d:self.K+2*self.K*self.d])

        assert (sigma > 0.).all(), "Sigma should be positive"
        assert (p > 0.).all(), "P should be positive"
        assert p.sum()>0.99 and p.sum()<1.01, "Sum of P should be equal to 1"

        # concatenates the paremeters and returns them as one single tensor
        return torch.cat([p, mu, sigma])
    
    def compute_log_det_jacobian_p(self, zeta_p:torch.Tensor):
        jacobian = torch.zeros(self.K, self.K)
        exp_sum = zeta_p.exp().sum() 
        for i in range(self.K):
            for j in range(self.K):
                if i != j:
                    jacobian[i,j] = - (zeta_p[i]+zeta_p[j]).exp() / exp_sum**2
                else:
                    jacobian[i,j] = zeta_p[i].exp()*(1 - zeta_p[i]/exp_sum) / exp_sum
        return torch.logdet(jacobian)
        
    def log_det(self, zeta:torch.Tensor):
        """Returns the log determinant of the jacobian of the inverse transformation"""
        sigmas_sum = torch.sum(zeta[self.K+self.K*self.d:self.K+2*self.K*self.d])
        log_det_jacobian_p = self.compute_log_det_jacobian_p(zeta[:self.K])
        return sigmas_sum + log_det_jacobian_p
    
    def log_prob(self, x:torch.Tensor, theta:torch.Tensor, full_data_size:int):
        """Returns the log probability of x under the distribution of the model
        
        Args:
            x: torch.Tensor, data
            theta: torch.Tensor, model parameters
            full_data_size: int, size of the full data set (used for batch learning)"""
        # gets the model parameters from the tensor params
        p = theta[:self.K]
        mu = theta[self.K:self.K+self.K*self.d].view(self.K,self.d)
        sigma = theta[self.K+self.K*self.d:self.K+2*self.K*self.d].view(self.K,self.d)
        assert(sigma > 0.).all() # Debugging

        # Sums (and scales for batch learning) the log probabilities of each point
        sum = 0.
        for i in range(x.shape[0]):
            l_prob = torch.zeros(self.K)
            for k in range(self.K):
                l_prob[k] = torch.log(p[k]) + Normal(mu[k], sigma[k]).log_prob(x[i]).sum()
            sum = sum + torch.logsumexp(torch.Tensor(l_prob), 0)

        sum = sum*full_data_size/x.shape[0]

        # print(sum.requires_grad)

        # print(f"sum: {sum}")
        sum += Normal(loc=0, scale=1).log_prob(torch.Tensor(mu)).sum()
        # print(f"Normal prior: {0.01*Normal(loc=0, scale=1).log_prob(torch.Tensor(mu)).sum()}")
        sum += Dirichlet(concentration=torch.ones(self.K)*self.alpha_0).log_prob(p)
        # print(f"Dirichlet prior: {0.01*Dirichlet(concentration=torch.ones(self.K)*self.alpha_0).log_prob(p)}")
        sum += LogNormal(loc=0, scale=1).log_prob(torch.Tensor(sigma)).sum()
        # print(f"LogNormal prior: {0.01*LogNormal(loc=0, scale=1).log_prob(torch.Tensor(sigma)).sum()}")

        return sum
   