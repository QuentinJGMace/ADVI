import numpy as np
import matplotlib.pyplot as plt
import torch

from advi2 import ADVI2
from ppca import PPCA_model, PPCA_with_ARD_model

# note that in the function below D should be greater then K
def build_toy_dataset(N, D, K, sigma=1):
    x_train = np.zeros([D, N])
    w = np.zeros([D,K])
    for k in range(K):
        w[k,k]=1.0/(k+1)
        w[k+1,k]=-1.0/(k+1)
    print(w)
    z = np.random.normal(0.0, 1.0, size=(K, N))
    mean = np.dot(w, z)
    shift=np.zeros([D])
    shift[1]=10
    for d in range(D):
      for n in range(N):
        x_train[d, n] = np.random.normal(mean[d, n], sigma)+shift[d]
    print("True principal axes:")
    print(w)
    print("Shift:")
    print(shift)
    return x_train.astype(np.float32,copy=False)


#ed.set_seed(142)

N = 1000  # number of data points
D = 2  # data dimensionality
K = 1  # latent dimensionality

# DATA

x_train = build_toy_dataset(N, D, K, sigma = 0.1)
x_train = torch.tensor(x_train).permute(1,0)

model = PPCA_with_ARD_model(D, 2)
param_keys = model.named_params
param_dims = model.dim_parameters
key_pos = model.key_pos
advi = ADVI2(model, 1, batch_size=10, lr=0.01, mode='meanfield', num_epochs=20)

advi.fit(x_train, method="Adam", plotting=True)
W = advi.model_params.vparams[0, :2].view(2, 1).detach()
mean = advi.model_params.vparams[0, 2:4].detach()
log_sigma = advi.model_params.vparams[0, 4:6].detach()
log_alpha = advi.model_params.vparams[0, 6:7].detach()
samples = model.rsample(50, W, mean, torch.exp(log_sigma), torch.exp(log_alpha))
plt.scatter(x_train[:, 0], x_train[:, 1], alpha=0.5)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.show()