import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal, Gamma
import numpy as np

class ModelParam(nn.Module):
    def __init__(self, dim, mode="meanfield"):
        self.dim = dim
        self.mode = mode

        if self.mode not in ["meanfield"]:
            raise ValueError("mode should be either 'meanfield''")
        
        # Init the variational parameters
        self.vparams = None
        if self.mode == "meanfield":
            self.mean = torch.zeros(dim)
            self.log_std = torch.zeros(dim)

            self.vparams = torch.stack([self.mean, self.log_std])
        # elif self.mode == "fullrank":
        #     self.mean = torch.zeros(dim)
        #     self.L = torch.eye(dim)

        #     self.vparams = torch.stack([self.mean, self.L.view(-1)])

        self.vparams.requires_grad = True

        self.size = self.vparams.size(0)
    
    def dist(self):
        if self.mode == "meanfield":
            return Normal(self.mean, self.log_std.exp())
        # elif self.mode == "fullrank":
        #     return MultivariateNormal(self.mean, self.L @ self.L.t())
        
    def rsample(self, n=1):
        return self.dist().rsample(n)
    
    def log_q(self, x):
        return self.dist().log_prob(x).sum()
        # elif self.mode == "fullrank":
        #     return MultivariateNormal(self.mean, self.L @ self.L.t()).log_prob(x).sum()
        


class ADVI:
    def __init__(self, model, inv_T, num_samples=1, lr=0.01, max_iter=1000, mode="fullrank"):
        self.model = model
        self.inv_T = inv_T
        self.delta_elbo_threshold = 1e-6
        self.num_samples = num_samples
        self.max_iter = max_iter
        self.lr = lr
        self.mode = mode
        self.model_params = {}

        if mode not in ["meanfield", "fullrank"]:
            raise ValueError("mode should be either 'meanfield' or 'fullrank'")
        
        if self.mode == "meanfield":
            self.model.init_meanfield()
        elif self.mode == "fullrank":
            raise NotImplementedError("Fullrank not implemented yet")
            self.model.init_fullrank()

    def init_meanfield(self):
        self.model_params = {'mu' : ModelParam(self.model.d, mode="meanfield"),
                             'log_sigma' : ModelParam(self.model.d, mode="meanfield")
                            }
        
    def log_prior(self, real_params, params):
        """
        The log prior of the parameters
        real_params: the parameters (in R) of the model
        params: same as real_params but in their orignal space"""
        if self.mode == "meanfield":
            log_mu = Normal(0, 1).log_prob(params['mu']).sum()
            log_sigma = Gamma(1, 1).log_prob(params['sigma']).sum()

    def det_J(self, real_params, params):
        """
        log(|det(Jacobian_T-1)|) of the transformation from the real space to the original space"""
        if self.mode == "meanfield":
            return params['log_sigma'].sum()
        
    def entropy(self, real_params, params): #Up to a constant since we only need the gradient
        """The entropy of the model"""
        if self.mode == "meanfield":
            return params['log_sigma'].sum()
        elif self.mode == "fullrank":
            raise NotImplementedError("Fullrank not implemented yet")
        
    def elbo(self, data):
        """The ELBO of the model"""
        real_params = {}
        for key in self.model_params:
            real_params[key] = self.model_params[key].rsample(1)
        
        if self.mode == "meanfield":
            params = {'mu': real_params['mu'],
                      'log_sigma': real_params['log_sigma'].exp()}
        elif self.mode == "fullrank":
            raise NotImplementedError("Fullrank not implemented yet")
            params = {'mu': real_params['mu'],
                      'L': real_params['L']}
            
        log_prior = self.log_prior(real_params, params)
        log_q = self.log_q(real_params, params)
        log_likelihood = self.model.log_prob(data, params)
        entropy = self.entropy(real_params, params)
        det_J = self.det_J(real_params, params)

        elbo = log_likelihood + det_J + entropy
        return elbo
        
        
    def log_q(self, real_params, model_params: ModelParam):
        out = 0.0
        for key in model_params:
            out += model_params[key].log_q(real_params[key])
        return out
        
    def init_fullrank(self):
        self.model_params = {'mu' : ModelParam(self.model.d, mode="meanfield"),
                             'L': ModelParam(self.model.d*(self.model.d + 1)/2, mode="meanfield")
                            }
    
    def fit(self, X: torch.Tensor):
        """Fits the parameters using ADVI onto the given data X
        
        Parameters:
        X (torch.Tensor): The data to fit the parameters on"""
        self.model.train() # Activates the gradient computation
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) # Not used in the paper, we could find an other one
        if self.mode == "meanfield":
            self.fit_meanfield(X, optimizer)
        elif self.mode == "fullrank":
            self.fit_fullrank(X, optimizer)

    def fit_meanfield(self, X: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Fits the parameters using ADVI with meanfield approximation onto the given data X
        
        Parameters:
        X (torch.Tensor): The data to fit the parameters on
        optimizer (torch.optim.Optimizer): The optimizer to use"""

        elbo, delta_elbo = 0, np.inf
        i = 0
        while i < self.max_iter and delta_elbo > self.delta_elbo_threshold:
            optimizer.zero_grad()
            z = torch.randn(self.num_samples, self.model.num_params)
            elbo = self.model.elbo(X, z, self.model.params)
            loss = -elbo #Optimisers are for minimisation
            loss.backward()
            optimizer.step()
            i += 1
    
    def fit_fullrank(self, X: torch.Tensor, optimiser: torch.optim.Optimizer):
        raise NotImplementedError("Fullrank not implemented yet")