# A standard inn model
import pickle

import numpy as np

import torch
import torch.optim as optim
from dmatch.models.nn.dense_nets import MLP

from nflows import flows

from tensorboardX import SummaryWriter

import sys
from pathlib import Path

from dmatch.utils import hyperparams
from dmatch.utils.hyperparams import get_dist, get_measure
from dmatch.utils.plotting import plot_slice, plot_coloured
from dmatch.utils.training import fit_generator

sys.path.append(str(Path('.').absolute().parent))

# from utils.load_mnist import load_mnist
from dmatch.models.flow_models import flow_builder
from dmatch.models.nn.flows import autoregessive, coupling_spline

from dmatch.utils.post_process import post_process_plane, get_ood, get_samples_model
from dmatch.utils.io import get_top_dir
from dmatch.utils.MC_estimators import get_kl_and_error

from dmatch.data.data_loaders import load_plane_dataset, data_handler

# Ideally this would be implemented as follows
# Currently only runs an autoencoder with no distribution matching
import argparse

parser = argparse.ArgumentParser()

# Dataset parameters
parser.add_argument('--dataset', type=str, default='checkerboard', help='The dataset to train on.')
# Currently this is not implemented, but it is a useful feature.
parser.add_argument('-d', type=str, default='INN_test', help='Directory to save contents into.')
parser.add_argument('-n', type=str, default='test_implicit', help='The name with which to tag saved outputs.')
parser.add_argument('--ndata', type=int, default=int(1e5), help='The number of data points to generate.')
parser.add_argument('--latent_dim', type=int, default=2, help='The data dimension.')

## Hyper parameters
parser.add_argument('--spline', type=int, default=0, help='Whether or not to use a spline transformation.')
parser.add_argument('--batch_size', type=int, default=100, help='Size of batch for training.')
parser.add_argument('--epochs', type=int, default=5,
                    help='The number of epochs to train for.')
parser.add_argument('--base_dist', type=str, default='normal',
                    help='A string to index the corresponding nflows distribution.')
parser.add_argument('--nstack', type=int, default=4,
                    help='The number of spline transformations to stack in the inn.')
parser.add_argument('--nblocks', type=int, default=2,
                    help='The number of layers in the networks in each spline transformation.')
parser.add_argument('--nodes', type=int, default=128,
                    help='The number of nodes in each of the neural spline layers.')
parser.add_argument('--nbins', type=int, default=30,
                    help='The number of bins to use in each spline transformation.')
parser.add_argument('--activ', type=str, default='leaky_relu',
                    help='The activation function to use in the networks in the neural spline.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='The learning rate.')
parser.add_argument('--reduce_lr_plat', type=int, default=0,
                    help='Whether to apply the reduce learning rate on plateau scheduler.')
parser.add_argument('--gclip', type=int, default=5,
                    help='The value to clip the gradient by.')

## Saving
parser.add_argument('--get_kl', type=int, default=0, help='Integer whether to calculate the KL divergence or not.')
parser.add_argument('--get_sinkhorn', type=int, default=0,
                    help='Integer whether to calculate the KL divergence or not.')
parser.add_argument('--get_ood', type=int, default=0,
                    help='Integer whether to calculate the fraction of OOD samples or not.')
parser.add_argument('--final_plot_encoding', type=int, default=1, help='Generate and save samples?')
parser.add_argument('--load', type=int, default=0, help='Generate and save samples?')

## reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

## KL estimate
parser.add_argument('--nrun', type=int, default=2,
                    help='The number of MC KL estimates to calculate.')
parser.add_argument('--nrun_sink', type=int, default=50,
                    help='The number of sinkhorn estimates to calculate.')
parser.add_argument('--ncalc', type=int, default=int(1e5),
                    help='The number of samples to pass through the encoder per sample.')
parser.add_argument('--n_test', type=int, default=10,
                    help='The number of times to calculate ncalc samples.')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Hyper params from passed args
bsize = args.batch_size
n_epochs = args.epochs
exp_name = args.n

sv_dir = get_top_dir()
log_dir = sv_dir + '/logs/' + exp_name
writer = SummaryWriter(log_dir=log_dir)

# with open(sv_dir + '/images' + '/' + args.d + '/exp_info_{}.png'.format(exp_name), 'wb') as f:
#     pickle.dump(vars(args), f)

inp_dim = args.latent_dim

# Set all tensors to be created on gpu, this must be done after dataset creation
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
print(device)

# Make dataset
# ndata = args.ndata
# dim = None if not args.dataset == 'hypercheckerboard' else args.latent_dim
# trainset = load_plane_dataset(args.dataset, ndata, dim=dim)
trainset = data_handler(args.ndata, bsize, inp_dim, args.dataset, device)

# Set up base transformation
# If using a normal distribution you have to allow for the possibility of samples coming from outside of the tail bound
bdist_shift = None
if args.base_dist == 'uniform':
    tail_bound = 1.
    if trainset.bounded:
        tails = None
    else:
        bdist_shift = tail_bound
        tails = 'linear'
if args.base_dist == 'normal':
    tail_bound = 4.
    if trainset.bounded:
        tails = None
    else:
        bdist_shift = tail_bound
        tails = 'linear'
    # Scale the data to be at the tail bounds
    # trainset.data *= tail_bound
    trainset.scale = tail_bound
transformation = autoregessive(inp_dim, args.nodes, num_blocks=args.nblocks, nstack=args.nstack, tail_bound=tail_bound,
                               tails=tails, activation=hyperparams.activations[args.activ], num_bins=args.nbins,
                               spline=args.spline)
base_dist = hyperparams.nflows_dists(args.base_dist, inp_dim, shift=bdist_shift, bound=tail_bound)
flow = flows.Flow(transformation, base_dist)

# Build model
flow_model = flow_builder(flow, base_dist, device, exp_name, directory=args.d)

# Define optimizers and learning rate schedulers
optimizer = optim.Adam(flow.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.ndata / bsize * n_epochs, 0)

# Reduce lr on plateau at end of epochs
if args.reduce_lr_plat:
    reduce_lr_inn = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')]
else:
    reduce_lr_inn = None

# Fit the model
if args.load:
    model_name = get_top_dir() + '/experiments/data/saved_models/model_{}'.format(flow_model.exp_name)
else:
    fit_generator(flow_model, optimizer, trainset, n_epochs, bsize, writer, schedulers=scheduler,
                  schedulers_epoch_end=reduce_lr_inn, gclip=args.gclip)


if args.final_plot_encoding:
    flow_model.flow.eval()
    if args.spline:
        model_name = 'NSF'
    else:
        model_name = 'RealNVP'
    get_samples_model(flow_model, trainset, model_name, args.dataset, device)

# Generate test data and preprocess etc
if args.latent_dim == 2:
    # Plotting the colored test
    testset = trainset.sample(int(1e5))

    import matplotlib.pyplot as plt

    fig, ax_ = plt.subplots(1, 2, figsize=(12, 5))
    ax = fig.axes
    plot_coloured(testset, testset, ax[0], 'Base distribution', args.dataset, 0.05)

    with torch.no_grad():
        target = flow_model(testset)
    plot_coloured(testset, target, ax[1], 'Prediction', args.dataset, 0.05)
    fig.tight_layout()
    fig.savefig(sv_dir + '/images' + '/' + args.d + '/coloured_{}.png'.format(exp_name))

    dim = None if not args.dataset == 'hypercheckerboard' else args.latent_dim
    # testset = load_plane_dataset(args.dataset, int(1e5), dim=dim)
    # if args.base_dist == 'normal':
    #     testset.data *= tail_bound
    bnd = tail_bound + 0.5
    post_process_plane(flow_model, testset, invertible=True, implicit=False, sup_title=args.dataset + ' INN',
                       bounds=[-bnd, bnd])

# With this you sample (n_calc * n_test) number of samples and calculate the kl divergence, This is repeated nrun times
if args.get_kl:
    nrun = args.nrun
    n_calc = args.ncalc
    n_test = args.n_test

    flow_model.flow.eval()
    kl_div_info = get_kl_and_error(get_dist(args.dataset, 2), get_dist(args.base_dist, 2), flow_model.encode, n_calc,
                                   nrun, n_test, device)

    print('Estimated KL divergence of {} with variance {}'.format(kl_div_info[0], kl_div_info[1]))

    with open(sv_dir + '/images' + '/' + flow_model.dir + '/score_{}.npy'.format(exp_name), 'wb') as f:
        np.save(f, kl_div_info)

if args.get_sinkhorn:
    args.nrun_sink = 1000
    dist_measure = get_measure('sinkhorn')
    btest = 1000
    sv = np.zeros(args.nrun_sink)
    for i in range(args.nrun_sink):
        normal_samples = base_dist.sample(btest)
        data_obj = data_handler(btest, btest, inp_dim, args.dataset, device)
        if args.base_dist == 'normal':
            data_obj.data *= tail_bound
        with torch.no_grad():
            encoded_samples = flow_model.encode(data_obj.data[0])
        sv[i] = dist_measure(normal_samples, encoded_samples)

    print(f'Sinkhorn Distance {np.mean(sv):.6f}')

    import matplotlib.pyplot as plt
    from dmatch.utils import plot2Dhist

    cum_mean = np.cumsum(sv) / np.arange(1, args.nrun_sink + 1)
    plt.figure()
    plt.plot(cum_mean[10:])
    plt.savefig(sv_dir + '/images' + '/' + flow_model.dir + f'/mns_plot{exp_name}')

    with open(sv_dir + '/images' + '/' + flow_model.dir + '/sinkhorn_{}.npy'.format(exp_name), 'wb') as f:
        np.save(f, sv)

# Calculate the number of OOD samples
if args.get_ood:
    nsamples = int(1e5)
    nrun = args.nrun

    bound = 4
    nbins = 50

    percent_ood, percent_oob, counts = get_ood(flow_model, nsamples, nrun, bound, nbins, max_it=1)

    nm = sv_dir + '/images' + '/' + flow_model.dir + '/slice_{}'.format(exp_name + '_') + '{}.png'''
    plot_slice(counts, nm.format('pred'))

    with open(sv_dir + '/images' + '/' + flow_model.dir + '/ood_{}.npy'.format(exp_name), 'wb') as f:
        np.save(f, percent_ood)
        np.save(f, percent_oob)
