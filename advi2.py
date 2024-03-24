import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class ModelParam2(nn.Module):

    def __init__(self, size, mode="meanfield"):
        self.size = size
        self.mode = mode

        if self.mode not in ["meanfield"]:
            raise ValueError("mode should be either 'meanfield''")
        
        # Init the variational parameters
        self.vparams = None
        if self.mode == "meanfield":
            self.mean = torch.zeros(size)
            self.log_std = torch.zeros(size)

            #concatenates the mean and the log_std into a 1D tensor
            self.vparams = torch.stack([self.mean, self.log_std])
        # elif self.mode == "fullrank":
        #     self.mean = torch.zeros(size)
        #     self.L = torch.eye(size)

        #     self.vparams = torch.stack([self.mean, self.L.view(-1)])

        self.vparams.requires_grad = True

        self.size = self.vparams.size(0)

    def dist(self):
        if self.mode == "meanfield":
            return Normal(self.vparams[0], self.vparams[1].exp() + 1e-6)
        # elif self.mode == "fullrank":
        #     return MultivariateNormal(self.mean, self.L @ self.L.t())
    
    def rsample(self, n=1):
        return self.dist().rsample(n)
    
    def log_q(self, x):
        return self.dist().log_prob(x).sum()
    
class ADVI2:

    def __init__(self, model, num_samples=1, batch_size = 10, lr=0.01, num_epochs=100, mode="meanfield"):

        self.model = model
        self.param_keys = self.model.named_params
        self.param_dims = self.model.dim_parameters
        self.advi_dim = 0
        self.batch_size = batch_size
        for key in self.param_keys:
            self.advi_dim += self.param_dims[key]

        self.key_pos = self.model.key_pos

        self.delta_elbo_threshold = 1e-6

        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.lr = lr
        self.mode = mode
        self.model_params = {}

        if mode not in ["meanfield", "fullrank"]:
            raise ValueError("mode should be either 'meanfield' or 'fullrank'")
        
        if self.mode == "meanfield":
            self.init_meanfield()
        elif self.mode == "fullrank":
            raise NotImplementedError("Fullrank mode not implemented yet")
            self.init_fullrank()

        self.elbo_history = []
        # print(self.model_params.vparams[1])

    def init_meanfield(self):
        self.model_params = ModelParam2(self.advi_dim, mode="meanfield")
    
    def grad_p_invT_logdet(self, x, sample, full_data_size):
        # Computes the term in common for all the gradients involved in ADVI
        with torch.no_grad():
            zeta = self.zeta_from_sample(sample)
            theta = self.model.theta_from_zeta(zeta)

        # zeros the entire gradient graph
        zeta.requires_grad = True
        theta.requires_grad = True

        log_prob = self.model.log_prob(x, theta, full_data_size)
        # print(log_prob)
        # Computes the gradient of log_prob w.r.t. theta
        log_prob.backward()
        grad_log_prob = theta.grad

        # Computes the gradient of the inverse transformation w.r.t. zeta
        grad_invT = self.model.grad_inv_T(zeta)

        # Computes the gradient of the log determinant of the jacobian of the inverse transformation w.r.t. zeta
        log_det = self.model.log_det(zeta)
        log_det.backward()
        grad_log_det = zeta.grad

        total = grad_log_prob*grad_invT + grad_log_det
        total.requires_grad = False
        # clips the gradient to avoid numerical instability
        total = torch.clamp(total, -10, 10)
        #print(total.abs().mean())
        return total

    def zeta_from_sample(self, sample):
        if self.mode == "meanfield":
            diag_exp = torch.exp(self.model_params.vparams[1])
            return diag_exp * sample + self.model_params.vparams[0]
        elif self.mode == "fullrank":
            raise NotImplementedError("Fullrank mode not implemented yet")

    def entropy(self):
        return self.model_params.dist().entropy().sum()

    def compute_elbo(self, batch, full_data_size):
        elbo = 0
        for i in range(self.num_samples):
            sample = torch.randn(self.advi_dim)
            zeta = self.zeta_from_sample(sample)  
            print(f"Zeta: {zeta}")
            theta = self.model.theta_from_zeta(zeta)
            print(f"Theta: {theta}")
            print(f"Log prob: {self.model.log_prob(batch, theta, full_data_size)}")
            print(f"Log det: {self.model.log_det(zeta)}")
            print(f"Entropy: {self.entropy()}")
            elbo += self.model.log_prob(batch, theta, full_data_size) + self.model.log_det(zeta) + self.entropy()

        elbo /= self.num_samples
        return elbo
    
    def update_params(self, grad_mu, grad_log_std, method="SGD"):
        if method == "SGD":
            self.model_params.vparams[0] += self.lr * grad_mu
            self.model_params.vparams[1] += self.lr * grad_log_std
        else:
            raise NotImplementedError("Only SGD is implemented for now")
    
    def fit(self, x, method = "SGD", plotting=True):

        if method == "SGD":
            optimizer = torch.optim.SGD([self.model_params.vparams], lr=self.lr)
        elif method == "Adam":
            optimizer = torch.optim.Adam([self.model_params.vparams], lr=self.lr)
        else:
            raise ValueError("Method should be either 'SGD' or 'Adam'")
        
        diff_elbo = np.inf
        last_elbo = -np.inf

        counter = 0
        elbo_history = []
        grad_mus = []
        grad_log_stds = []

        pbar = tqdm(total=self.num_epochs)
        pbar.set_description("Fitting...")

        # Main loop
        while counter < self.num_epochs and diff_elbo > self.delta_elbo_threshold:

            for i_batch in range(0, x.size(0), self.batch_size):
                grad_mu = torch.zeros(self.advi_dim)
                grad_log_std = torch.zeros(self.advi_dim)
                batch = x[torch.randint(0, x.size(0), (self.batch_size,))]
                # for i in range(self.num_samples):
                #     sample = torch.randn(self.advi_dim)
                    # commun_grad = self.grad_p_invT_logdet(batch, sample, len(x))
                    # commun_grad.requires_grad = False
                    # grad_mu += commun_grad
                    # if self.mode == "meanfield":
                    #     diag_exp = torch.diag(torch.exp(self.model_params.vparams[1]))
                    #     # print("Diag exp 1", diag_exp)
                    #     grad_log_std_tmp = torch.mm((commun_grad * sample).view(1, self.advi_dim), diag_exp) + 1
                    #     grad_log_std_tmp = grad_log_std_tmp.view(self.advi_dim)
                    #     grad_log_std += grad_log_std_tmp
                optimizer.zero_grad()
                elbo = self.compute_elbo(batch, x.shape[0])
                loss = -elbo
                print(f"Loss: {loss}")
                loss.backward()
                self.model_params.vparams.grad.clamp_(-10, 10)
                optimizer.step()

                # # Average the gradients
                # grad_mu /= self.num_samples
                # grad_log_std /= self.num_samples
                # grad_mu.requires_grad = False
                # grad_log_std.requires_grad = False

                # grad_mus.append(grad_mu)
                # if self.mode == "meanfield":
                #     grad_log_stds.append(grad_log_std)

                # # Update the parameters
                # self.update_params(grad_mu, grad_log_std, method)

            # Compute the ELBO
            try:
                with torch.no_grad():
                    elbo = self.compute_elbo(x, x.shape[0])
                #elbo = self.compute_elbo(x)
            except ValueError:
                elbo = -np.inf

            # Store the results
            elbo_history.append(elbo)

            diff_elbo = np.abs(elbo - last_elbo)
            if np.isnan(diff_elbo):
                diff_elbo = np.inf
            last_elbo = elbo

            counter += 1
            pbar.update(1)

        pbar.close()
        if plotting:
            plt.plot([i for i in range(len(elbo_history))], elbo_history)
            plt.show()