# This script is for matching an N-dimensional normal distribution to a 2D plane dataset. (N=latent_dim)
# Train using a supervised loss with targets assigned using an inverse box muller transformation
import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

from dmatch.data.hyper_dim import HyperCheckerboardDataset
from dmatch.models.nn.dense_nets import MLP, SplineNet
from dmatch.utils import hyperparams

from dmatch.utils.hyperparams import get_dist
from dmatch.utils.io import get_top_dir
from dmatch.utils.plotting import plot_slice
from dmatch.utils.post_process import get_ood

### Set the seeds for reproducibility
torch.seed(42)
np.random.seed(42)

import argparse

#### Define arguments to pass from command line ########################################################################

parser = argparse.ArgumentParser()

## Saving
parser.add_argument('-d', '--outputdir', type=str, default='supervised_ND_ND',
                    help='Choose the base output directory')
parser.add_argument('-n', '--outputname', type=str, default='test_',
                    help='Set the output name directory')
parser.add_argument('--plt', type=int, default=1, help='Integer whether to plot training and distributions.')
parser.add_argument('--get_kl', type=int, default=0, help='Integer whether to calculate the KL divergence or not.')

## Base distribution arguments
parser.add_argument('--latent_dim', type=int, default=2, help='The dimension of the base distribution.')

## Dataset
parser.add_argument('--nsteps_train', type=int, default=100,
                    help='The number of batches per epoch to train on, this with the batch size defines the size of the dataset.')
parser.add_argument('--nsteps_val', type=int, default=10,
                    help='The number of batches per epoch to use for validation.')

## Model parameters
parser.add_argument('--model', type=str, default='splinet',
                    help='The type of model to sue during training.')
parser.add_argument('--dropout', type=float, default=0,
                    help='A global parameter for controlling the dropout between layers.')
parser.add_argument('--activation', type=str, default='none',
                    help='The activation function to apply to the output of the "encoder".')
parser.add_argument('--bnorm', type=int, default=0,
                    help='An integer specifying whether to apply batch normalization to the model.')
parser.add_argument('--lnorm', type=int, default=0,
                    help='An integer specifying whether to apply layer normalization to the model.')
parser.add_argument('--sw', type=int, default=512, help='Set number of nodes.')
parser.add_argument('--sd', type=int, default=3, help='Set depth.')

## Training parameters
parser.add_argument('--batch_size', type=int, default=1000, help='Size of batch for training.')
parser.add_argument('--epochs', type=int, default=10,
                    help='The number of epochs to train for.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='The learning rate.')
parser.add_argument('--wd', type=float, default=0.01,
                    help='The weight decay parameter to use in the AdamW optimizer.')

## KL estimate
parser.add_argument('--nrun', type=int, default=2,
                    help='The number of estimates to calculate.')
parser.add_argument('--ncalc', type=int, default=int(1e5),
                    help='The number of samples to pass through the encoder per sample.')

#### Collect arguments and begin script ################################################################################
args = parser.parse_args()

### Define the distributions
latent_dim = args.latent_dim
output_dim = latent_dim
base_dist = get_dist('indp_gauss', latent_dim)
target_dist = get_dist('checkerboard', output_dim)

### Get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

### Define the model to train
layers_outer = [args.sw] * args.sd
out_activ = hyperparams.activations[args.activation]
drop_p = args.dropout
if args.model == 'dense':
    model = MLP(latent_dim, output_dim, layers=layers_outer, output_activ=out_activ, drp=drop_p,
                batch_norm=args.bnorm,
                layer_norm=args.lnorm).to(device)
elif args.model == 'splinet':
    model = SplineNet(latent_dim, output_dim, layers=layers_outer, output_activ=out_activ, ).to(device)
else:
    raise NotImplementedError('No model with name {}'.format(args.model))

### Training definitions
nsteps_train = args.nsteps_train
nsteps_val = args.nsteps_val
batch_size = args.batch_size
n_epochs = args.epochs

### Optimizer and scheduler
# optimizers = [optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)]
optimizer = optim.Adam(model.parameters(), lr=args.lr)
max_step = nsteps_train * n_epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step, 0)
# scheduler = None

reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

### Train the model
monitor_interval = 100

exp_name = args.outputname + '_' + str(latent_dim)
top_dir = get_top_dir()
sv_dir = top_dir + '/images' + '/' + args.outputdir + '/'
if not os.path.exists(sv_dir):
    os.makedirs(sv_dir)

tbar = trange(n_epochs, position=0)
trec = trange(nsteps_train, desc='{desc}', position=2)
tval = trange(n_epochs, desc='{desc}', position=3)
train_save = []
val_save = []


# TODO: tidy this away somewhere
def inv_box_muller(data, shft=0):
    return torch.unsqueeze(torch.exp(-0.5 * (data[:, 0 + shft] ** 2 + data[:, 1 + shft] ** 2)), 1)


def inv_angles(data, shft=0):
    z0 = data[:, 0 + shft]
    z1 = data[:, 1 + shft]
    norm = (z0 ** 2 + z1 ** 2) ** (1 / 2)
    # Solve for the angle
    angles = torch.atan2(z1 / norm, z0 / norm)
    angles[angles < 0] += 2 * np.pi
    return angles / (2 * np.pi)


# TODO: This should be a method of data_handler
def get_target(data):
    # Now the input data will be an N-dimensional checkerboard and the output an N-dimensional checkerboard
    if data.shape[1] % 2:
        raise ValueError('The data dimension msut be a factor of two')
    # First get all N uniform distributions in (0, 1)
    axis_list = []
    for i in range(0, latent_dim):
        if not i % 2:
            axis_list += [inv_box_muller(data, shft=i - i % 2).view(-1)]
        else:
            axis_list += [inv_angles(data, shft=i - i % 2)]
    target = HyperCheckerboardDataset.split_cube(torch.stack(axis_list))

    return target


class data_handler():
    def __init__(self, nsample, batch_size):
        self.nsample = nsample
        self.batch_size = batch_size
        self.update_data()

    def update_data(self):
        self.data = base_dist.sample([self.nsample]).to(device).view(-1, self.batch_size, latent_dim)

    def get_loss(self, i):
        # On the start of each epoch generate new samples, and then for each proceeding epoch iterate through the data
        if i == 0:
            self.update_data()
        encoding = model(self.data[i])
        target = get_target(self.data[i])
        loss = nn.MSELoss()(target, encoding)
        return loss


loss_obj = data_handler(batch_size * nsteps_train, batch_size)

for epoch in tbar:
    # Training
    running_loss = []
    for i in range(nsteps_train):
        # zero the parameter gradients before calculating the losses
        optimizer.zero_grad()
        loss = loss_obj.get_loss(i)
        loss.backward(loss.backward(retain_graph=True))
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i % monitor_interval == 0:
            running_loss += [loss.item()]
            s = '[{}, {}] {}'.format(epoch + 1, i + 1, running_loss[-1])
            trec.set_description_str(s)

    # Update training loss trackers
    train_save += [np.mean(running_loss)]

    # Validation
    val_loss = []
    for i in range(nsteps_val):
        val_loss += [loss_obj.get_loss(i).item()]

    val_save += [np.mean(val_loss, 0)]

    if reduce_lr:
        reduce_lr.step(val_save[-1])

    torch.save(model.state_dict(), sv_dir + '/model_{}'.format(exp_name))

### Evaluate
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('There are {} parameters'.format(nparams))
model.eval()

# How many of the first epochs to ignore for plotting
mx = 5

### Plotting
if args.plt:
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.plot(val_save[mx:], label='validation')
    ax.plot(train_save[mx:], '--', label='test')
    ax.legend()
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    fig.savefig(sv_dir + '/training_{}.png'.format(exp_name))

### Get the fraction of out of distribution samples
nsamples = args.ncalc
nrun = args.nrun

bound = 4
nbins = 50

nbatch = 10
data_generator = lambda nsamples: data_handler(int(nsamples / nbatch), nbatch).data
percent_ood, percent_oob, counts, counts_true = get_ood(model, nsamples, nrun, bound, nbins, data_generator, get_target,
                                                        max_it=100)

nm = sv_dir + '/slice_{}'.format(exp_name + '_') + '{}.png'''
plot_slice(counts, nm.format('pred'))
plot_slice(counts_true, nm.format('truth'))

with open(sv_dir + '/ood_{}.npy'.format(exp_name), 'wb') as f:
    np.save(f, percent_ood)
    np.save(f, percent_oob)
