# Calculate and plot the salience of a trained model with respect to different loss terms
# Demonstrates the similarity between the likelihood of the transformation under the prior and the sinkhorn loss
import os

import torch
import matplotlib.pyplot as plt

from dmatch.data.plane import RotatedCheckerboardDataset, CheckerboardDataset
from dmatch.models.flow_models import flow_builder
from dmatch.models.nn.flows import autoregessive

from dmatch.utils import hyperparams

from dmatch.utils.hyperparams import get_dist, get_measure
from dmatch.utils.io import get_top_dir

from nflows import flows

import argparse

#### Define arguments to pass from command line ########################################################################
from dmatch.utils.plotting import plot_salience, plot_coloured
from dmatch.utils.post_process import get_salience

parser = argparse.ArgumentParser()

## Saving
parser.add_argument('-d', '--outputdir', type=str, default='salience',
                    help='Choose the base output directory')
parser.add_argument('-n', '--outputname', type=str, default='test',
                    help='Set the output name directory')
parser.add_argument('--get_mle', type=int, default=0,
                    help='Whether to calculate the MLE derivatives or not.')
parser.add_argument('--get_sinkhorn', type=int, default=0,
                    help='Whether to calculate the sinkhorn derivatives or not.')
parser.add_argument('--plot_colored', type=int, default=1,
                    help='Whether to plot colored encodings or not.')

## Base distribution arguments
parser.add_argument('--base_dist', type=str, default='checkerboard',
                    help='A string to index the corresponding torch distribution.')
parser.add_argument('--latent_dim', type=int, default=2, help='The dimension of the base distribution.')
parser.add_argument('--batch_size', type=int, default=5000, help='The batch size.')

## Dataset
parser.add_argument('--target', type=str, default='normal',
                    help='A string to index the corresponding plane dataset distribution.')
parser.add_argument('--output_dim', type=int, default=2, help='The output dimension.')
parser.add_argument('--n_sample', type=int, default=int(1e6),
                    help='The number of samples to calculate the saliency for.')

## Model parameters
parser.add_argument('--model', type=str, default='nsf',
                    help='The type of model to use during training.')
parser.add_argument('--dist_measure', type=str, default='sinkhorn',
                    help='The distribution matching loss to compare against.')

#### Collect arguments and begin script ################################################################################
args = parser.parse_args()

### Define the distributions
latent_dim = args.latent_dim

base_dist = get_dist(args.base_dist, latent_dim)
output_dim = args.output_dim
target_dist = get_dist(args.target, output_dim)

exp_name = args.outputname + '_' + str(latent_dim)
top_dir = get_top_dir()
sv_dir = top_dir + '/images' + '/' + args.outputdir
if not os.path.exists(sv_dir):
    os.makedirs(sv_dir, exist_ok=True)

### Get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transformation = autoregessive(2, 128, num_blocks=2, nstack=4,
                               tail_bound=4.0,
                               tails=None, activation=hyperparams.activations['leaky_relu'], num_bins=30,
                               spline=1)
base_dist_flow = hyperparams.nflows_dists('normal', 2, shift=None, bound=4.0)
flow = flows.Flow(transformation, base_dist_flow)
flow_model = flow_builder(flow, base_dist_flow, device, exp_name, directory='dummy')
flow_model.load(get_top_dir() + f'/experiments/data/saved_models/model_flows_paper_{args.base_dist}',
                device=device)
flow_model.eval()
flow_model = flow_model.to(device)

if args.plot_colored:
    nsample = int(1e5)
    filler_alpha = 0.01
    filler = RotatedCheckerboardDataset(nsample, flip_axes=True).data.to(device)
    data = CheckerboardDataset(nsample, flip_axes=True).data.to(device)

    with torch.no_grad():
        target = flow_model.encode(data)
        target_fill = flow_model.encode(filler)

    # Plot colored encodings of the different models so that the mapping of the mass can be visualised.
    fig, ax_ = plt.subplots(1, 2, figsize=(10, 5))
    ax = fig.axes
    inp = data.view(-1, args.latent_dim)
    plot_coloured(inp, inp, ax[0], 'Base distribution', args.base_dist, set_limits=False, filler=filler,
                  filler_alpha=filler_alpha)
    plot_coloured(inp, target, ax[1], 'Prediction', args.base_dist, set_limits=False, filler=target_fill,
                  filler_alpha=filler_alpha)
    fig.tight_layout()
    fig.savefig(sv_dir + '/coloured_{}.png'.format(exp_name))

if args.get_mle:
    x = torch.distributions.uniform.Uniform(torch.zeros(latent_dim).to(device) - 4.,
                                            torch.ones(latent_dim).to(device) * 4.,
                                            validate_args=None).sample([args.n_sample])

    flow_batch_size = int(1e5)
    n_run = int(args.n_sample // flow_batch_size)

    data = x.view(n_run, flow_batch_size, latent_dim)
    y = torch.zeros_like(data)
    saliency = torch.zeros((n_run, flow_batch_size))
    saliency_detJ = torch.zeros_like(saliency)
    saliency_likelihood = torch.zeros_like(saliency)

    for i in range(n_run):
        # Get the model output
        x = data[i]

        x.requires_grad_()
        if x.grad is not None:
            x.grad.zero_()

        # Get the total flow loss
        scores = flow_model.log_prob(x)
        scores.backward(torch.ones_like(scores))
        saliency[i] = x.grad.norm(dim=1)

        # Get the derivatives of the determinant of the Jacobian
        x.grad.zero_()
        y[i], detJ = flow_model.flow._transform(x)
        detJ.backward(torch.ones_like(detJ))
        saliency_detJ[i] = x.grad.norm(dim=1)

        # Get the derivatives of the data likelihood
        x.grad.zero_()
        likelihood = flow_model.base_dist.log_prob(flow_model.encode(x))
        likelihood.backward(torch.ones_like(likelihood))
        saliency_likelihood[i] = x.grad.norm(dim=1)

    saliency = saliency.view(-1)
    saliency_detJ = saliency_detJ.view(-1)
    saliency_likelihood = saliency_likelihood.view(-1)
    y = y.view(-1, latent_dim)
    x = data.view(-1, latent_dim)

    plot_salience(x, y, saliency, sv_dir + f'/salience_{exp_name}.png', n_bins=200)
    plot_salience(x, y, saliency_detJ, sv_dir + f'/salience_detJ_{exp_name}.png', n_bins=200)
    plot_salience(x, y, saliency_likelihood, sv_dir + f'/salience_likelihood_{exp_name}.png', n_bins=200)

    print('Plotted MLE')

if args.get_sinkhorn:
    batch_size = args.batch_size
    dist_measure = get_measure(args.dist_measure)

    uniform_sample = torch.distributions.uniform.Uniform(torch.zeros(latent_dim).to(device) - 4.,
                                                         torch.ones(latent_dim).to(device) * 4.,
                                                         validate_args=None).sample([args.n_sample])

    # get_salience(uniform_sample, 'uniform', flow_model, args.n_sample, batch_size, target_dist, device, dist_measure,
    #              sv_dir, exp_name)
    nsamp = args.n_sample
    r_checkers = RotatedCheckerboardDataset(nsamp, flip_axes=True).data.to(device)
    ood, r_y, ood_salience = get_salience(r_checkers, 'rotated_checkers', flow_model, nsamp, batch_size, target_dist,
                                          device, dist_measure, sv_dir, exp_name)

    checkers = CheckerboardDataset(nsamp, flip_axes=True).data.to(device)
    ind, y, ind_salience = get_salience(checkers, 'checkers', flow_model, nsamp, batch_size, target_dist, device,
                                        dist_measure, sv_dir, exp_name)

    x = torch.cat((ood, ind))
    y = torch.cat((r_y, y))
    salience = torch.cat((ood_salience, ind_salience))[:, 0]
    plot_salience(x, y, salience, sv_dir + f'/salience_joined_{exp_name}.png', n_bins=200)

    print('Finished job.')
