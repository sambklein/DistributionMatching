import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# Borrowing from nflows package
from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform

from .flows import autoregessive, coupling_spline
from nflows.utils import torchutils

# TODO: there is a lot of code duplication in this file - fix this.
from .splines import SPLASH, NeuralSplash


def norm_exception():
    raise Exception("Can't set both layer and batch normalization")


class ResnetBlock(nn.Module):
    def __init__(self, input_dim, forward_layers=[300, 300, 300]):
        super(ResnetBlock, self).__init__()
        self.input_dim = input_dim

        # Build the networks forward layers
        forward_layers = deepcopy(forward_layers)
        forward_layers += [input_dim]
        self.forward_functions = nn.ModuleList([nn.Linear(input_dim, forward_layers[0])])
        self.forward_functions.extend(nn.ModuleList(
            [nn.Linear(forward_layers[i], forward_layers[i + 1]) for i in range(len(forward_layers) - 1)]))

    def forward(self, inputs):
        # Forward transform of the MLP
        x = inputs
        for i, function in enumerate(self.forward_functions[:-1]):
            x = function(x)
            x = F.relu(x)
        x = self.forward_functions[-1](x)
        return x + inputs


class SkipNet(nn.Module):
    def __init__(self, input_dim, latent_dim, output_activ=nn.Identity(), num_blocks=3, skip_dim=32, resnet_depth=3):
        super(SkipNet, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_activ = output_activ

        self.functions = nn.ModuleList([nn.Linear(input_dim, skip_dim)])
        self.functions.extend(
            nn.ModuleList(
                [ResnetBlock(skip_dim, forward_layers=[skip_dim] * resnet_depth) for _ in range(num_blocks)]
            )
        )
        self.functions.extend(nn.ModuleList([nn.Linear(skip_dim, latent_dim)]))

    def forward(self, x):
        for block in self.blocks[:-1]:
            x = block(x)
            x = F.relu(x)
        x = self.output_activ(self.functions[-1](x))
        return x


class combine_models(nn.Module):
    def __init__(self, *args):
        super(combine_models, self).__init__()
        self.functions = nn.ModuleList([])
        try:
            [self.functions.extend(model.functions) for model in args]
        except:
            [self.functions.extend([model]) for model in args]
        # self.model = nn.Sequential(*args)
        self.models = args

    def forward(self, x, **kwargs):
        for model in self.models:
            x = model(x, **kwargs)
        return x


def maker(input_dim, output_dim):
    return MLP(input_dim, output_dim, layers=[64, 64, 64])


class NsfOT(nn.Module):
    def __init__(self, input_dim, nsplines=10, coupling=False, activation=F.relu, tails=None, spline=True,
                 tail_bound=4., output_activ=None):
        super(NsfOT, self).__init__()
        self.output_activ = output_activ
        if coupling:
            self.transform = coupling_spline(input_dim, maker, nstack=3, tail_bound=tail_bound, tails=tails, lu=0,
                                             num_bins=nsplines)
        else:
            self.transform = autoregessive(input_dim, 128, nstack=4, num_bins=nsplines, tail_bound=tail_bound,
                                           tails=tails,
                                           activation=activation, spline=spline)

    def forward(self, x, get_det=0, **kwargs):
        encoding, log_det = self.transform(x)
        if self.output_activ is not None:
            encoding = self.output_activ(encoding)
        if get_det:
            return encoding, log_det
        else:
            return encoding


class MLP(nn.Module):
    def __init__(self, input_dim, latent_dim, islast=True, output_activ=nn.Identity(), layers=[300, 300, 300], drp=0,
                 batch_norm=False, layer_norm=False, int_activ=torch.relu, bias=True, inst_norm=False,
                 change_init=True):
        super(MLP, self).__init__()
        # Very lazy
        layers = deepcopy(layers)
        self.latent_dim = latent_dim
        self.drp_p = drp
        self.inner_activ = int_activ

        self.functions = nn.ModuleList([nn.Linear(input_dim, layers[0], bias=bias)])
        if islast:
            layers += [latent_dim]
        self.functions.extend(
            nn.ModuleList([nn.Linear(layers[i], layers[i + 1], bias=bias) for i in range(len(layers) - 1)]))
        if change_init:
            # Change the initilization
            for function in self.functions:
                torch.nn.init.xavier_uniform_(function.weight)
                if bias:
                    function.bias.data.fill_(0.0)
        self.output_activ = output_activ

        if batch_norm and layer_norm:
            norm_exception()

        self.norm = 0
        self.norm_func = nn.LayerNorm
        if batch_norm:
            self.norm = 1
            self.norm_func = nn.BatchNorm1d
        if layer_norm:
            self.norm = 1
            self.norm_func = nn.LayerNorm
        self.norm_funcs = nn.ModuleList([self.norm_func(layers[i]) for i in range(len(layers) - 1)])
        self.inst_norm = inst_norm
        if inst_norm:
            self.inst_funcs = nn.ModuleList([nn.InstanceNorm1d(1) for _ in range(len(layers) - 1)])

    def forward(self, x, context=None, **kwargs):
        for i, function in enumerate(self.functions[:-1]):
            x = function(x)
            if self.norm:
                x = self.norm_funcs[i](x)
            if self.inst_norm:
                x = self.inst_funcs[i](x.view(x.shape[0], 1, x.shape[1])).view(*x.shape)
            x = self.inner_activ(x)
            x = nn.Dropout(p=self.drp_p)(x)
        x = self.output_activ(self.functions[-1](x))
        return x

    def batch_predict(self, data_array, encode=False):
        store = []
        for data in data_array:
            if encode:
                store += [torch.cat(self.encode(data), 1)]
            else:
                store += [self(data)]
        return torch.cat(store)


class StochasticMLP(MLP):
    def __init__(self, input_dim, latent_dim, islast=True, output_activ=nn.Identity(), layers=[300, 300, 300], drp=0,
                 batch_norm=False, layer_norm=False, vae=False):
        super(StochasticMLP, self).__init__(input_dim, latent_dim, islast=islast, output_activ=output_activ,
                                            layers=layers, drp=drp, batch_norm=batch_norm, layer_norm=layer_norm)
        self.vae = vae
        self.functions.extend(nn.ModuleList([nn.Linear(layers[-1], latent_dim)]))

    def encode(self, x):
        for i, function in enumerate(self.functions[:-2]):
            x = function(x)
            if self.norm:
                x = self.norm_funcs[i](x)
            x = F.relu(x)
            x = nn.Dropout(p=self.drp_p)(x)
        mu = self.output_activ(self.functions[-2](x))
        logvar = self.output_activ(self.functions[-1](x))
        return mu, logvar

    def forward(self, x, get_values=None, **kwargs):
        if get_values is None:
            get_values = self.vae
        mu, logvar = self.encode(x)
        if get_values:
            return mu, logvar
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu


class SplineNet(MLP):
    def __init__(self, input_dim, latent_dim, islast=True, output_activ=nn.Identity(), layers=[300, 300, 300], drp=0,
                 nsplines=100, init='linear', spline_type='linear', final_only=False, nstop=4,
                 batch_norm=False, layer_norm=False
                 ):
        super(SplineNet, self).__init__(input_dim, latent_dim, islast=islast, output_activ=output_activ,
                                        layers=layers, drp=drp, batch_norm=batch_norm, layer_norm=layer_norm)
        if spline_type == 'splash':
            self.activs = nn.ModuleList(
                [SPLASH(layers[i], nknots=nsplines, init=init) for i in range(len(self.functions[:-1]))])
        else:
            self.activs = nn.ModuleList(
                [NeuralSplash(layers[i], nknots=nsplines, init=init) for i in range(len(self.functions[:-1]))])
        self.nstop = nstop
        self.final_only = final_only

    def forward(self, x, **kwargs):
        for i, function in enumerate(self.functions[:-1]):
            x = function(x)
            if (self.nstop < i) and (not self.final_only):
                x = self.activs[i](x)
            elif (i == len(self.functions[:-1]) - 1) and self.final_only:
                x = self.activs[i](x)
            else:
                x = torch.relu(x)
        x = self.output_activ(self.functions[-1](x))
        return x
