# An autoencoder with latent space matching applied directly to the output of the first encoder, or implicitly using
# Dense nets.

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from nflows import distributions

from tensorboardX import SummaryWriter

# import sys
# from pathlib import Path
#
# sys.path.append(str(Path('.').absolute().parent))

from dmatch.models.nn.dense_nets import MLP, StochasticMLP, dense_decoder, dense_inner
from dmatch.utils import get_measure, hyperparams
from dmatch.models import implicit_autoencoder
from dmatch.models import standard_autoencoder

# from utils.load_mnist import load_mnist
from dmatch.utils import fit
from dmatch.utils import post_process_plane

from dmatch.data.data_loaders import load_plane_dataset
# from data.plane import CheckerboardDataset

torch.manual_seed(4)
np.random.seed(4)

bsize = 1000
n_epochs = 10
# TODO: Add possibility of dim(A) != dim(Z)
latent_dim = 2
out_activ = hyperparams.activations['none']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

ndata = int(1e5)
dataset = load_plane_dataset('checkerboard', ndata, npad=0, scale=False)
dataset.data = (dataset.data + 4) / 4 - 1

trainloader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=True, num_workers=1)

inp_dim = 2
layers = [256, 256, 128, 128]

stochastic = 1

if stochastic:
    encoder = StochasticMLP(inp_dim, latent_dim, layers=layers, output_activ=out_activ, drp=0,
                            batch_norm=0, layer_norm=0).to(device)
else:
    encoder = MLP(inp_dim, latent_dim, layers=layers, output_activ=out_activ, drp=0,
                  batch_norm=0, layer_norm=0).to(device)

# Define optimizers and learning rate schedulers
max_step = ndata / bsize * n_epochs
optimizer_outer = optim.AdamW(encoder.parameters(), lr=0.001, weight_decay=0.01)
scheduler_outer = optim.lr_scheduler.CosineAnnealingLR(optimizer_outer, max_step, 0)
# scheduler_outer = optim.lr_scheduler.StepLR(optimizer_outer, 10, 0.1)

# Define the Model
dist_measure = get_measure('sinkhorn')
base_dist = hyperparams.torch_dists('uniform', latent_dim)

for epoch in range(n_epochs):
    running_loss = []
    for i, data in enumerate(trainloader, 0):
        data = data.to(device)
        noise = base_dist.sample([bsize])
        loss = dist_measure(encoder(data), noise.to(device))
        loss.backward()
        optimizer_outer.step()
        # scheduler_outer.step()
        if i % 100 == 99:
            print('[{}, {}] loss: {}'.format(epoch, i, loss.item()))

testset = load_plane_dataset(args.dataset, int(1e5), npad=pwidth, scale=False)
testset.data = (testset.data + 4) / 4 - 1
samples = encoder(testset.data.to(device))
plt.figure()
plt.hist2d(samples[:, 0], samples[:, 1], bins=100)
plt.savefig('../images/plane_samples.png')
