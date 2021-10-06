import torch
import torch.nn as nn
import torch.distributions as torch_distributions
import nflows
from geomloss import SamplesLoss
from torch.nn import functional as F
import torch.optim as optim

from dmatch.data.data_loaders import load_plane_dataset
from dmatch.data.hyper_dim import SparseHyperCheckerboardDataset, HyperCheckerboardDataset, HyperSpheres, HyperSphere, \
    HyperCross, HyperSouthernCross, HyperShells
from dmatch.utils.io import on_cluster


def my_relu(x):
    return F.relu6(x) / 2 - 1


# TODO: why is this not a function?
activations = {
    'none': nn.Identity(),
    'relu': F.relu,
    'elu': F.elu,
    'leaky_relu': F.leaky_relu,
    'relu6': F.relu6,
    'relu1': my_relu,
    'sigmoid': F.sigmoid,
    'tanh': torch.tanh,
    'hard_tanh': nn.Hardtanh(),
    'gelu': nn.GELU(),
    'celu': nn.CELU(),
    'selu': nn.SELU(),
    'rrelu': nn.RReLU(),
    'prelu': nn.PReLU(),
    'softmax': nn.Softmax()
}


def get_optimizer(optimizer, parameters, lr, wd=0.01, momentum=0.9):
    if optimizer == 'AdamW':
        return optim.AdamW(parameters, lr=lr, weight_decay=wd)
    if optimizer == 'Adam':
        return optim.Adam(parameters, lr=lr)
    if (optimizer == 'sgd') or (optimizer == 'SGD'):
        return optim.SGD(parameters, lr=lr, momentum=momentum)


recon_losses = {
    'none': nn.Identity(),
    'mse': nn.MSELoss(),
    'l1': nn.L1Loss(),
    'bce': nn.BCELoss(),
    'sinkhorn': SamplesLoss('sinkhorn', scaling=0.5, blur=0.01, p=1)
}

# A wrapper for nflows distributions that will discard values outside of bound, assumes out of bounds values are unlikely
# Bound is a single value outside of which no samples are drawn
# TODO: this should be made a proper class in the nflows package.
from nflows.distributions.base import Distribution


class rejection_sampler(Distribution):
    def __init__(self, dist, bound=None):
        super().__init__()
        self.sampler = dist
        self.bound = bound

    def sample_with_rejection(self, num):
        sample = self.sampler.sample(num + 1000)
        sample = sample[torch.all((-self.bound < sample) & (sample < self.bound), 1)]
        sample = sample[:num]
        return sample

    def _sample(self, num, context):
        if self.bound:
            # TODO: this should be a while loop or something, for now it isn't important
            sample = self.sample_with_rejection(num)
        else:
            sample = self.sampler.sample(num)
        return sample

    def _log_prob(self, inputs, context):
        return self.sampler._log_prob(inputs, context)

    def _mean(self, context):
        return self.sampler._mean(context)


def nflows_dists(nm, inp_dim, shift=False, bound=None, device='cpu'):
    try:
        tshift = 0
        bshift = 0
        if shift:
            tshift = shift
            bshift = shift - 1
        return {
            'uniform': nflows.distributions.uniform.BoxUniform(torch.zeros(inp_dim).to(device) - tshift,
                                                               torch.ones(inp_dim).to(device) + bshift),
            'normal': rejection_sampler(nflows.distributions.StandardNormal([inp_dim]), bound)
            # 'normal': nflows.distributions.StandardNormal([inp_dim])
        }[nm]

    except KeyError:
        raise ValueError('Unknown nflows base distribution: {}'.format(nm))


class indp_gaussians():
    def __init__(self, dim):
        self.latent_dim = dim
        self.sampler = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def sample(self, ndata):
        samples = []
        for _ in range(self.latent_dim):
            samples += [self.sampler.sample(ndata)]
        return torch.stack(samples)


def torch_dists(nm, latent_dim, device='cpu'):
    try:
        return {
            'uniform': torch_distributions.uniform.Uniform(torch.zeros(latent_dim).to(device) - 1,
                                                           torch.ones(latent_dim).to(device),
                                                           validate_args=None),
            'normal': torch_distributions.MultivariateNormal(torch.zeros(latent_dim).to(device),
                                                             torch.eye(latent_dim).to(device)),
            'indp_gauss': indp_gaussians(latent_dim)
        }[nm]

    except KeyError:
        raise ValueError('Unknown torch base distribution: {}'.format(nm))


def get_distances(x, y):
    norms_x = torch.sum(torch.square(x), axis=1, keepdims=True)
    norms_y = torch.sum(torch.square(y), axis=1, keepdims=True)
    dot_prod = torch.matmul(x, torch.transpose(y, 0, 1))
    return norms_x + torch.transpose(norms_y, 0, 1) - 2. * dot_prod


def kernel_sum(x, y, from_self=True):
    # If you switch from using a normal dist the one needs to change to the standard deviation of the new dist
    C = 2 * x.shape[1] * torch.tensor(0.1)
    distance = get_distances(x, y)
    if from_self:
        distance *= 1 - torch.eye(x.shape[0])
    return torch.sum(C / (C + distance))


def compute_MMD(x, y):
    xds = kernel_sum(x, x)
    yds = kernel_sum(y, y)
    x_y_ds = kernel_sum(x, y, from_self=False)
    n = x.shape[0]
    return 1 / (n * (n - 1)) * (xds + yds) - 2 / (n ** 2) * x_y_ds


def get_measure(name):
    if name == 'None' or name == 'none':
        def dist(x, y):
            return torch.tensor(0)

    elif name == 'sinkhorn':
        dist = SamplesLoss('sinkhorn', scaling=0.95, blur=0.01)

    elif name == 'sinkhorn_fast':
        dist = SamplesLoss('sinkhorn', scaling=0.8, blur=0.01)

    elif name == 'sinkhorn1':
        dist = SamplesLoss('sinkhorn', scaling=0.5, blur=0.01, p=1)

    elif name == 'mmd':
        dist = compute_MMD

    elif name == 'rbf':
        dist = SamplesLoss('gaussian')

    else:
        raise NotImplementedError(f'No distance with name {name}')

    def dist_measure(x, y):
        return dist(x, y)

    return dist_measure


class sampler():
    def __init__(self, name):
        self.name = name

    def sample(self, ndata):
        data_obj = load_plane_dataset(self.name, ndata[0])
        return data_obj.data


def get_dist(name, dim, std=0.05, device='cpu'):
    try:
        dist = torch_dists(name, dim, device=device)
    except:
        if name == 'checkerboard':
            dist = HyperCheckerboardDataset(int(1e3), dim)
        elif name == 'sparse_checkerboard':
            dist = SparseHyperCheckerboardDataset(int(1e3), dim)
        elif name == 'nspheres':
            dist = HyperSpheres(int(1e3), dim)
        elif name == 'cross':
            dist = HyperCross(int(1e3), dim)
        elif name == 'stars':
            dist = HyperSouthernCross(int(1e3), dim, std=std)
        elif name == 'sphere':
            dist = HyperSphere(int(1e3), dim)
        elif name == 'shells':
            dist = HyperShells(int(1e3), dim, std=std)
        else:
            dist = sampler(name)
            dist.sample([8])
    return dist
