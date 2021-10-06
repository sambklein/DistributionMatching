# This script is for matching an N-dimensional normal distribution to a 2D plane dataset. (N=latent_dim)
import os

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

from dmatch.utils import hyperparams

from dmatch.models.nn.dense_nets import MLP, NsfOT, dense_decoder
from dmatch.utils.MC_estimators import get_kl_and_error

from dmatch.utils.hyperparams import get_measure, get_dist
from dmatch.utils.io import get_top_dir
from dmatch.utils.plotting import plot2Dhist, plot_coloured, projectiontionLS_2D
import pickle

### Set the seeds for reproducibility
# TODO: Need to set the seeds in all other scripts
# TODO: you need to set this properly
# torch.seed(42)
# np.random.seed(42)

import argparse

#### Define arguments to pass from command line ########################################################################
from dmatch.utils.training import get_log_det

parser = argparse.ArgumentParser()

## Saving
parser.add_argument('-d', '--outputdir', type=str, default='AAE',
                    help='Choose the base output directory')
parser.add_argument('-n', '--outputname', type=str, default='test',
                    help='Set the output name directory')
parser.add_argument('--plt', type=int, default=1, help='Integer whether to plot training and distributions.')
parser.add_argument('--get_kl', type=int, default=0, help='Integer whether to calculate the KL divergence or not.')

## Loading
parser.add_argument('--load', type=int, default=0, help='Whether to load existing experiment.')

## Base distribution arguments
parser.add_argument('--base_dist', type=str, default='checkerboard',
                    help='A string to index the corresponding torch distribution.')
parser.add_argument('--std', type=float, default=0.05,
                    help='Standard deviation defined in corresponding dataset (only for stars at present).')
parser.add_argument('--latent_dim', type=int, default=2,
                    help='The dimension of the base distribution.')
parser.add_argument('--noise_strength', type=float, default=0,
                    help='The number to multiply the standard normal that is added to the sample data by.')

## Dataset
parser.add_argument('--nsteps_train', type=int, default=100,
                    help='The number of batches per epoch to train on, this with the batch size defines the size of '
                         'the dataset.')
parser.add_argument('--nsteps_val', type=int, default=10,
                    help='The number of batches per epoch to use for validation.')
parser.add_argument('--target', type=str, default='normal',
                    help='A string to index the corresponding plane dataset distribution.')
parser.add_argument('--target_dim', type=int, default=2,
                    help='The dimension of the output.')
parser.add_argument('--chain_dims', type=int, default=0,
                    help='Set the latent dimension equal to the input dimension.')

## Model parameters
parser.add_argument('--model', type=str, default='dense',
                    help='The type of model to use during training.')
parser.add_argument('--dropout', type=float, default=0,
                    help='A global parameter for controlling the dropout between layers.')
parser.add_argument('--activation', type=str, default='none',
                    help='The activation function to apply to the output of the "encoder".')
parser.add_argument('--inner_activ', type=str, default='selu',
                    help='The activation function to apply between layers of the "encoder".')
parser.add_argument('--bnorm', type=int, default=0,
                    help='An integer specifying whether to apply batch normalization to the model.')
parser.add_argument('--lnorm', type=int, default=0,
                    help='An integer specifying whether to apply layer normalization to the model.')
parser.add_argument('--nknots', type=int, default=8,
                    help='The number of knots to use.')
parser.add_argument('--sw', type=int, default=512, help='Set number of nodes.')
parser.add_argument('--sd', type=int, default=3, help='Set depth.')

# AAE settings
# Slurm hates bools for some reason
parser.add_argument('--wa', type=int, default=0, help='Train a wasserstein autoencoder?')
parser.add_argument('--t_disc', type=int, default=2,
                    help='The step on which to train the discriminator, if N, train on the Nth step.')
parser.add_argument('--aae_lr', type=float, default=0.0001,
                    help='Learning rate for the adversary.')
parser.add_argument('--aae_wd', type=float, default=0.01,
                    help='Weight decay for the adversary.')

## Training parameters
parser.add_argument('--batch_size', type=int, default=1000, help='Size of batch for training.')
parser.add_argument('--epochs', type=int, default=10,
                    help='The number of epochs to train for.')
parser.add_argument('--optim', type=str, default='Adam',
                    help='The optimizer to use for training.')
parser.add_argument('--scheduler', type=int, default=0,
                    help='Whether to use a learning rate scheduler.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='The learning rate.')
parser.add_argument('--wd', type=float, default=0.001,
                    help='The weight decay parameter to use in the AdamW optimizer.')
parser.add_argument('--p', type=float, default=0.9,
                    help='The momentum to use in SGD.')
parser.add_argument('--beta_j', type=float, default=0,
                    help='The number to multiply the log determinant in the loss by.')
parser.add_argument('--gclip', type=float, default=2,
                    help='Gradient clipping to apply.')
parser.add_argument('--train_ae', type=int, default=0,
                    help='The type of model to sue during training.')
parser.add_argument('--beta_dist', type=float, default=1,
                    help='The number to multiply the distribution matching in the AE by.')

## KL estimate
parser.add_argument('--nrun', type=int, default=2,
                    help='The number of MC KL estimates to calculate.')
parser.add_argument('--ncalc', type=int, default=int(1e5),
                    help='The number of samples to pass through the encoder per sample.')
parser.add_argument('--n_test', type=int, default=10,
                    help='The number of times to calculate ncalc samples.')

## Distribution measure
parser.add_argument('--dist_measure', type=str, default='AAE',
                    help='')

#### Collect arguments and begin script ################################################################################
args = parser.parse_args()

output_dim = args.target_dim
if args.chain_dims:
    latent_dim = output_dim
    args.latent_dim = output_dim
else:
    latent_dim = args.latent_dim

# Saving things
exp_name = args.outputname + '_' + str(latent_dim)
top_dir = get_top_dir()
sv_dir = top_dir + '/images' + '/' + args.outputdir + '/'
if not os.path.exists(sv_dir):
    os.makedirs(sv_dir, exist_ok=True)

# kwarg of whether to load a model or train a new model
ld = args.load
get_kl = args.get_kl
if ld:
    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)


    # with open(sv_dir + 'exp_info_{}.pkl.png'.format(exp_name), 'rb') as f:
    with open(sv_dir + 'exp_info_{}.pkl'.format(exp_name), 'rb') as f:
        args = Bunch(pickle.load(f))
else:
    with open(sv_dir + 'exp_info_{}.pkl'.format(exp_name), 'wb') as f:
        pickle.dump(vars(args), f)

base_dist = get_dist(args.base_dist, latent_dim)
target_dist = get_dist(args.target, output_dim)

### Get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

### Define the distribution matching term
# layers_aae = [128, 128, 64, 32, 16]
if args.model == 'dense':
    layers_aae = [512] * 3
else:
    layers_aae = [512] * 3
out_activ = torch.nn.Identity() if args.wa else torch.sigmoid
discriminator = MLP(output_dim, 1, layers=layers_aae, output_activ=torch.sigmoid, batch_norm=0).to(device)

### Define the model to train
layers_outer = [args.sw] * args.sd
out_activ = hyperparams.activations[args.activation]
int_activ = hyperparams.activations[args.inner_activ]
drop_p = args.dropout

if args.model == 'dense':
    model = MLP(latent_dim, output_dim, layers=layers_outer, output_activ=out_activ, drp=drop_p,
                batch_norm=args.bnorm,
                layer_norm=args.lnorm, int_activ=int_activ).to(device)
elif args.model == 'nsf':
    if args.base_dist == 'normal':
        tails = 'linear'
    else:
        tails = None
    if latent_dim != output_dim:
        raise Exception('The NSF only works on dimension preserving maps')
    model = NsfOT(latent_dim, nsplines=args.nknots, tails=tails, spline=1).to(device)

if args.train_ae:
    import torch.nn as nn

    decoder = dense_decoder(output_dim, latent_dim, layers=layers_outer, output_activ=nn.Identity()).to(device)

else:
    decoder = None

nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(nparams)

### Training definitions
batch_size = args.batch_size
nsteps_train = args.nsteps_train
nsteps_val = args.nsteps_val
n_epochs = args.epochs


### Optimizer and scheduler
def get_optimizer_schedulers(model, n_epochs, lr, wd, decoder=None):
    params = list(model.parameters())
    if decoder is not None:
        params += list(decoder.parameters())
    optimizer = hyperparams.get_optimizer(args.optim, model.parameters(), lr=lr, wd=wd)
    max_step = nsteps_train * n_epochs
    if args.scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step, 0)
    else:
        scheduler = None
    return optimizer, scheduler


if args.train_ae > 0:
    n_epochs_stage_one = args.train_ae
else:
    n_epochs_stage_one = n_epochs

optimizer, scheduler = get_optimizer_schedulers(model, n_epochs_stage_one, args.lr, args.wd,
                                                decoder=decoder)
optimizer_aae, scheduler_aae = get_optimizer_schedulers(discriminator, n_epochs_stage_one, args.aae_lr, args.aae_wd)
optimizers = [optimizer_aae, optimizer]
schedulers = [scheduler_aae, scheduler]

### Train the model
monitor_interval = 100

tbar = trange(n_epochs, position=0)
train_save = []
val_save = []

class data_handler():
    def __init__(self, batch_size, nsteps_train, nsteps_val=0):
        self.nsample = batch_size * nsteps_train
        self.nval = batch_size * nsteps_val
        self.batch_size = batch_size
        self.nsteps_train = nsteps_train
        self.nsteps_val = nsteps_val
        self.nsample_valid = batch_size * nsteps_val
        self.update_data()

    def update_data(self, valid=False):
        nsample = self.nsample_valid if valid else self.nsample
        self.data = base_dist.sample([nsample]).to(device).view(-1, self.batch_size, latent_dim)
        self.target_sample = target_dist.sample([nsample]).to(device).view(-1, self.batch_size, output_dim)

    def get_valid_loss(self):
        loss = [- torch.mean(torch.log(discriminator(sample)) + torch.log(1 - discriminator(encoding)))]
        loss += [torch.mean(torch.log(discriminator(encoding)))]
        return loss


loss_obj = data_handler(batch_size, nsteps_train, nsteps_val)

if ld:
    # model.load_state_dict(torch.load(sv_dir + 'model_{}.png'.format(exp_name), map_location=device))
    model.load_state_dict(torch.load(sv_dir + 'model_{}'.format(exp_name), map_location=device))
else:
    for epoch in tbar:
        # Training
        running_loss = []
        for i in range(loss_obj.nsteps_train):
            # zero the parameter gradients before calculating the losses
            [optimizer.zero_grad() for optimizer in optimizers]

            if i == 0:
                loss_obj.update_data(valid=False)

            # Discriminator
            data = loss_obj.data[i]
            if (i + 1) % args.t_disc:
                model.eval()
                discriminator.train()
                encoding = model(data)
                sample = loss_obj.target_sample[i]
                if args.wa:
                    # loss = - torch.mean(discriminator(sample) - discriminator(encoding))
                    loss = - discriminator(sample).mean() + discriminator(encoding).mean()
                else:
                    loss = - torch.mean(torch.log(discriminator(sample)) + torch.log(1 - discriminator(encoding)))
                loss.backward()
                if (args.gclip > 0) or args.wa:
                    norm_type = 'inf' if args.wa else 2.0
                    clip_grad_norm_(discriminator.parameters(), args.gclip, norm_type=norm_type)
                    # clip_grad_value_(discriminator.parameters(), args.gclip)
                optimizers[0].step()
                if args.scheduler:
                    schedulers[0].step()

            # Generator
            if not ((i + 1) % args.t_disc):
                model.train()
                encoding = model(data)
                if args.wa:
                    loss = - torch.mean(discriminator(encoding))
                else:
                    loss = - torch.mean(torch.log(discriminator(encoding)))
                if args.beta_j > 0:
                    loss -= get_log_det(data, model).mean()
                if epoch < args.train_ae:
                    reconstruction = decoder(encoding)
                    loss = nn.L1Loss()(data, reconstruction) + loss * args.beta_dist

                loss.backward()
                optimizers[1].step()
                if args.scheduler:
                    schedulers[1].step()

            dt = 1
            if (args.base_dist == 'hepmass') and (i == 0):
                dt = 0
            if (i % monitor_interval == 0) and dt:
                losses = loss_obj.get_valid_loss()
                running_loss += [[loss.item() for loss in losses]]
                # trec.set_description_str(s)

        if epoch == args.train_ae:
            optimizer, scheduler = get_optimizer_schedulers(model, n_epochs_stage_one, args.lr, args.wd)
            optimizer_aae, scheduler_aae = get_optimizer_schedulers(discriminator, n_epochs_stage_one, args.aae_lr,
                                                                    args.aae_wd)
            optimizers = [optimizer_aae, optimizer]
            schedulers = [scheduler_aae, scheduler]

        # Update training loss trackers
        train_save += [np.mean(running_loss, 0)]

        # Validation
        val_loss = np.zeros((loss_obj.nsteps_val, len(losses)))
        loss_obj.update_data(valid=True)
        for i in range(loss_obj.nsteps_val):
            val_loss[i] = [loss.item() for loss in loss_obj.get_valid_loss()]

        val_save += [np.mean(val_loss, 0)]

    torch.save(model.state_dict(), sv_dir + '/model_{}'.format(exp_name))

### Evaluate
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
model.eval()

nsamples = int(1e5)
nbatch = 10
data = data_handler(int(nsamples / nbatch), nbatch).data


def batch_predict(data_array, encode=False):
    store = []
    for data in data_array:
        if encode:
            store += [torch.cat(model.encode(data), 1)]
        else:
            store += [model(data)]
    return torch.cat(store)


### Plotting
if args.plt:
    fig, ax_ = plt.subplots(1, len(optimizers), figsize=(20, 5))
    titles = ['Discriminator', 'Generator']
    for j, ax in enumerate(fig.axes):
        ax.plot([val[j] for val in val_save], label='validation')
        ax.plot([train[j] for train in train_save], '--', label='test')
        ax.set_title(titles[j])
        ax.legend()
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
    fig.savefig(sv_dir + '/training_{}.png'.format(exp_name))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # data = base_dist.sample([int(1e5)]).to(device)
    # output = model(data)
    with torch.no_grad():
        output = batch_predict(data)
    target = output.detach().cpu().numpy()
    plot2Dhist(target, ax, 100)
    # ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
    #                labelleft=False)
    # lim = 3.5
    # ax.set_xlim([-lim, lim])
    # ax.set_ylim([-lim, lim])
    fig.tight_layout()
    fig.savefig(sv_dir + '/encoded_distribution_{}.png'.format(exp_name))

if get_kl:
    nrun = args.nrun
    n_calc = args.ncalc
    n_test = args.n_test

    kl_div_info = get_kl_and_error(base_dist, target_dist, model, n_calc, nrun, n_test, device, g_chi=1)

    print('Estimated KL divergence of {} with variance {}'.format(kl_div_info[0], kl_div_info[1]))
    print('Estimated chi squared of {} with p value {}'.format(kl_div_info[2], kl_div_info[3]))

    with open(sv_dir + '/score_{}.npy'.format(exp_name), 'wb') as f:
        np.save(f, kl_div_info)
