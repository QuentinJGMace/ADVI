import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, MultivariateNormal, Gamma
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

            #concatenates the mean and the log_std into a 1D tensor
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
    def __init__(self, model, param_keys, param_dims, key_pos, num_samples=1, lr=0.01, max_iter=1000, mode="fullrank"):
        self.model = model
        self.delta_elbo_threshold = 1e-6
        self.num_samples = num_samples
        self.max_iter = max_iter
        self.lr = lr
        self.mode = mode
        self.model_params = {}

        self.param_keys = param_keys
        self.param_dims = param_dims
        self.key_pos = key_pos

        if mode not in ["meanfield", "fullrank"]:
            raise ValueError("mode should be either 'meanfield' or 'fullrank'")
        
        if self.mode == "meanfield":
            self.init_meanfield()
        elif self.mode == "fullrank":
            raise NotImplementedError("Fullrank not implemented yet")
            self.init_fullrank()

        # print([self.model_params[key] for key in self.model_params.keys()])

    def init_meanfield(self):
        self.model_params = {}
        for i, key in enumerate(self.param_keys):
            self.model_params[key] = ModelParam(self.param_dims[key], mode="meanfield")
                
    def log_prior(self, real_params, params):
        """
        The log prior of the parameters
        real_params: the parameters (in R) of the model
        params: same as real_params but in their orignal space"""
        if self.mode == "meanfield":
            log_mu = Normal(0, 1).log_prob(params['mu']).sum()
            # Counter intuitive, params represent the exponentiated log sigma
            # (to keep the same keys with real_params)
            log_sigma = Gamma(1, 1).log_prob(params['log_sigma']).sum()

    # DEPEND DES PARAMETRES DU MODELE
    def det_J(self, real_params, key_pos:list):
        """
        log(|det(Jacobian_T-1)|) of the transformation from the real space to the original space"""
        sum = Tensor([0.])
        sum.requires_grad = True
        for key in self.key_pos:
            sum += real_params[key].sum()
        return sum
    
    def entropy(self, model_params): #Up to a constant since we only need the gradient
        """The entropy of the model"""
        if self.mode == "meanfield":
            # return real_params['log_sigma'].exp().sum()
            sum = Tensor([0.])
            sum.requires_grad = True
            for key in model_params.keys():
                sum += model_params[key].log_std.sum()
        elif self.mode == "fullrank":
            raise NotImplementedError("Fullrank not implemented yet")
        return sum
        
    def elbo(self, data):
        """The ELBO of the model"""
        real_params = {}
        # print("ELBO !")
        for key in self.model_params:
            real_params[key] = self.model_params[key].rsample([1])
            
        # log_prior = self.log_prior(real_params, params)
        log_q = self.log_q(real_params, self.model_params)
        log_likelihood = self.model.log_prob(data, real_params)
        entropy = self.entropy(self.model_params)
        det_J = self.det_J(real_params, key_pos=self.key_pos)
        
        elbo = log_likelihood + det_J + entropy
        elbo.requires_grad = True
        # with torch.no_grad():
        #     print("ELBO: ", elbo, "\nlog_prior: ", log_prior, "\nlog_q: ", log_q, "\nlog_likelihood: ", log_likelihood, "\nentropy: ", entropy, "\ndet_J: ", det_J)
        return elbo
        
        
    def log_q(self, real_params, model_params: ModelParam):
        out = 0.0
        for key in model_params:
            out += model_params[key].log_q(real_params[key])
        return out
        
    def init_fullrank(self):
        raise NotImplementedError("Fullrank not implemented yet")
    
    def fit(self, X: torch.Tensor, plotting:bool=False):
        """Fits the parameters using ADVI onto the given data X
        
        Parameters:
        X (torch.Tensor): The data to fit the parameters on"""
        self.model.train() # Activates the gradient computation
        optimizer = torch.optim.Adam([self.model_params[key].vparams for key in self.model_params.keys()], lr=self.lr) # Not used in the paper, we could find an other one
        if self.mode == "meanfield":
            self.fit_meanfield(X, optimizer, plotting)
        elif self.mode == "fullrank":
            self.fit_fullrank(X, optimizer, plotting)

    def fit_meanfield(self, X: torch.Tensor, optimizer: torch.optim.Optimizer, plotting:bool=False):
        """Fits the parameters using ADVI with meanfield approximation onto the given data X
        
        Parameters:
        X (torch.Tensor): The data to fit the parameters on
        optimizer (torch.optim.Optimizer): The optimizer to use"""

        elbo, delta_elbo = 0, np.inf
        i = 0
        bar = tqdm(total=self.max_iter)
        elbo_list = []
        while i < self.max_iter and delta_elbo > self.delta_elbo_threshold:
            optimizer.zero_grad()
            elbo = self.elbo(X)
            loss = -elbo #Optimisers are for minimisation
            loss.backward()
            optimizer.step()
            i += 1
            elbo_list.append(elbo.item())
            bar.update(1)
        if plotting:
            plt.plot(elbo_list)
            plt.show()
    
    def fit_fullrank(self, X: torch.Tensor, optimiser: torch.optim.Optimizer, plotting:bool=False):
        raise NotImplementedError("Fullrank not implemented yet")