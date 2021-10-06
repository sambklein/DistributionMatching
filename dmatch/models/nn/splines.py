import torch

# Borrowing from nflows package
from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.utils import torchutils
from torch import nn as nn
from torch.nn import functional as F
from nflows import transforms


class SPLASH(nn.Module):
    def __init__(self, input_shape, nknots=10, init='random', batch=True):
        super(SPLASH, self).__init__()
        if nknots % 2 == 0:
            nknots += 1
        if batch:
            self.norm = nn.BatchNorm1d(input_shape)
            tail_bound = 3.3
        else:
            self.norm = torch.tanh
            tail_bound = 1
        self.register_buffer('bins', torch.linspace(0, tail_bound, nknots // 2))
        # self.register_buffer('bins', torch.exp(torch.linspace(0, torch.log(torch.tensor(3.3)), nknots // 2) ** 2) - 1.)

        if init == 'random':
            self.pdf = nn.Parameter(torch.Tensor(nknots))
            self.pdf.data.uniform_(-0.1, 0.1)
        elif init == 'linear':
            self.tail_bound = 1
            self.pdf = nn.Parameter(torch.linspace(-self.tail_bound, self.tail_bound, nknots))
        elif init == 'relu':
            self.pdf = nn.Parameter(torch.cat((torch.ones(1), torch.zeros(nknots - 1))))
        else:
            raise NotImplementedError(f'Only "linear" and "random" initialization implemented, not "{init}".')

        self.bias = nn.Parameter(torch.Tensor(nknots))
        self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x, norm=True):
        # This allows the spline to be accessed directly for plotting
        if norm:
            x = self.norm(x)
        output = torch.zeros_like(x)
        # TODO vectorise this implementation
        for i, bin in enumerate(self.bins):
            output += self.pdf[i] * torch.relu(x - bin) - self.pdf[-(i + 1)] * torch.relu(-x - bin)
        return output


class MonotonicSPLASH(nn.Module):
    def __init__(self, input_shape, nknots=10, init='random'):
        super(SPLASH, self).__init__()
        if nknots % 2 == 0:
            nknots += 1
        self.norm = nn.BatchNorm1d(input_shape)
        # TODO make this quantiles of a normal
        self.register_buffer('bins', torch.linspace(0, 3.3, nknots // 2))

        if init == 'random':
            self.pdf = nn.Parameter(torch.Tensor(nknots))
            self.pdf.data.uniform_(-0.1, 0.1)
        elif init == 'linear':
            self.tail_bound = 1
            self.pdf = nn.Parameter(torch.linspace(-self.tail_bound, self.tail_bound, nknots))
        elif init == 'relu':
            self.pdf = nn.Parameter(torch.cat((torch.ones(1), torch.zeros(nknots - 1))))
        else:
            raise NotImplementedError(f'Only "linear" and "random" initialization implemented, not "{init}".')

    def forward(self, x, norm=True):
        # This allows the spline to be accessed directly for plotting
        if norm:
            x = self.norm(x)
        output = torch.zeros_like(x)
        # TODO vectorise this implementation
        cdf = F.softmax(self.pdf, dim=-1)
        for i, bin in enumerate(self.bins):
            output += cdf[i] * torch.relu(x - bin) - cdf[-(i + 1)] * torch.relu(-x - bin)
        return output


# Apply the binning based on the norm of the input
class GlobalSPLASH(nn.Module):
    def __init__(self, input_shape, nknots=10, init='random'):
        super(SPLASH, self).__init__()
        if nknots % 2 == 0:
            nknots += 1
        self.norm = nn.BatchNorm1d(input_shape)
        # TODO make this quantiles of a normal
        self.register_buffer('bins', torch.linspace(0, 10, nknots))
        # self.register_buffer('bins', torch.exp(torch.linspace(0, torch.log(torch.tensor(3.3)), nknots // 2) ** 2) - 1.)

        if init == 'random':
            self.pdf = nn.Parameter(torch.Tensor(nknots))
            self.pdf.data.uniform_(-0.1, 0.1)
        elif init == 'linear':
            self.tail_bound = 1
            self.pdf = nn.Parameter(torch.linspace(-self.tail_bound, self.tail_bound, nknots))
        elif init == 'relu':
            self.pdf = nn.Parameter(torch.cat((torch.ones(1), torch.zeros(nknots - 1))))
        else:
            raise NotImplementedError(f'Only "linear" and "random" initialization implemented, not "{init}".')

    def forward(self, x, norm=True):
        output = torch.zeros_like(x)
        norms = ((x ** 2).sum(1) ** 0.5).view(-1, 1)
        for i, bin in enumerate(self.bins):
            output += self.pdf[i] * (x - bin) * ((norms - bin) > 0)
        return output


# Quadratic SPLASH
class QuadraticSplash(nn.Module):
    def __init__(self, input_shape, nknots=10, init='random'):
        super(QuadraticSplash, self).__init__()
        self.norm = nn.BatchNorm1d(input_shape)
        self.register_buffer('bins', torch.linspace(0, 3.3, nknots // 2))

        if init == 'random':
            self.pdf = nn.Parameter(torch.Tensor(nknots))
            self.pdf.data.uniform_(-0.1, 0.1)
            self.dcpl = nn.Parameter(torch.Tensor(nknots))
            self.dcpl.data.uniform_(-0.1, 0.1)
        elif init == 'linear':
            self.tail_bound = 1
            self.pdf = nn.Parameter(torch.linspace(-self.tail_bound, self.tail_bound, nknots))
            self.dcpl = nn.Parameter(torch.linspace(-self.tail_bound, self.tail_bound, nknots))
        else:
            raise NotImplementedError(f'Only "linear" and "random" initialization implemented, not "{init}".')

    def forward(self, x, norm=True):
        # This allows the spline to be accessed directly for plotting
        if norm:
            x = self.norm(x)
        output = torch.zeros_like(x)
        # TODO vectorise this implementation
        for i, bin in enumerate(self.bins):
            output += self.pdf[i] * torch.relu(x - bin) + self.dcpl[i] * torch.relu(x - bin) ** 2
            if i > 0:
                output += self.pdf[-i] * torch.relu(-x - bin) + self.dcpl[-i] * torch.relu(-x - bin) ** 2
        return output


class NeuralSplash(nn.Module):
    def __init__(self, input_shape, nknots=10, layers=[5, 5, 5], init='random'):
        super(NeuralSplash, self).__init__()
        self.norm = nn.BatchNorm1d(input_shape)
        self.register_buffer('bins', torch.linspace(0, 3.3, nknots // 2))
        from dmatch.models.nn.dense_nets import MLP
        self.models = nn.ModuleList(
            [MLP(1, 1, layers=layers, output_activ=torch.tanh) for _ in range(nknots)])

    def forward(self, x, norm=True):
        # This allows the spline to be accessed directly for plotting
        if norm:
            x = self.norm(x)
        output = torch.zeros_like(x)
        # TODO vectorise this implementation
        for i, bin in enumerate(self.bins):
            output += self.models[i](x.view(-1, 1)).view(x.shape) * torch.relu(x - bin)
            if i > 0:
                output += self.models[-i](x.view(-1, 1)).view(x.shape) * torch.relu(-x - bin)
        return output


def linear_spline(cdf, bin_boundaries, x, left=0, right=1, constrained=False):
    lb = bin_boundaries[0, 0, 0]
    rb = bin_boundaries[0, 0, -1]

    inside_interval_mask = (x >= left) & (x <= right)

    cdf = cdf[inside_interval_mask]
    bin_boundaries = bin_boundaries[inside_interval_mask]

    outputs = torch.zeros_like(x)

    inputs = x[inside_interval_mask]

    inv_bin_idx = torchutils.searchsorted(bin_boundaries.clone(), inputs)

    slopes = (cdf[..., 1:] - cdf[..., :-1]) / (
            bin_boundaries[..., 1:] - bin_boundaries[..., :-1]
    )
    offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

    inv_bin_idx = inv_bin_idx.unsqueeze(-1)
    input_slopes = slopes.gather(-1, inv_bin_idx)[..., 0]
    input_offsets = offsets.gather(-1, inv_bin_idx)[..., 0]
    outputs[inside_interval_mask] = (inputs * input_slopes + input_offsets)

    if constrained:
        outside_interval_mask = ~inside_interval_mask
        outputs[outside_interval_mask] = x[outside_interval_mask]
    else:
        outputs[x < left] = x[x < left] * left + left * (1 - lb)
        outputs[x > right] = x[x > right] * right + right * (1 - rb)

    return outputs


def get_net(features, hidden_features, num_blocks, output_multiplier):
    from dmatch.models.nn.dense_nets import MLP
    return MLP(features, features * output_multiplier, layers=[hidden_features] * num_blocks,
               int_activ=torch.tanh)


class NonInvertibleNSF(MaskedPiecewiseRationalQuadraticAutoregressiveTransform):

    def __init__(self, features, hidden_features, num_blocks=2, num_bins=10, net_only=1, **kwargs):
        super(NonInvertibleNSF, self).__init__(features, hidden_features, num_blocks=2, num_bins=num_bins, **kwargs)
        self.autoregressive_net = get_net(features, hidden_features, num_blocks, self._output_dim_multiplier())
        self.net_only = net_only

    def forward(self, inputs, context=None):
        enc, log_det = super(NonInvertibleNSF, self).forward(inputs, context=context)
        if self.net_only:
            return enc
        else:
            return enc, log_det


class get_net(nn.Module):

    def __init__(self, features, hidden_features, num_blocks, output_multiplier):
        super(get_net, self).__init__()
        from dmatch.models.nn.dense_nets import MLP
        self.feature_list = list(range(1, features + 1))
        self.makers = nn.ModuleList(
            [MLP(self.feature_list[i], output_multiplier, layers=[hidden_features] * num_blocks,
                 output_activ=nn.Identity()) for i in range(features)])

    def forward(self, data, context=None):
        splines = []
        for i, function in enumerate(self.makers):
            splines += [function(data[:, :self.feature_list[i]])]
        return torch.cat(splines, 1)


class PureNonInvertibleNSF(MaskedPiecewiseRationalQuadraticAutoregressiveTransform):

    def __init__(self, features, hidden_features, num_blocks=2, num_bins=10, **kwargs):
        super(PureNonInvertibleNSF, self).__init__(features, hidden_features, num_blocks=2, num_bins=num_bins, **kwargs)
        self.autoregressive_net = get_net(features, hidden_features, num_blocks, self._output_dim_multiplier())


class StackedNonInvertibleNSF(nn.Module):

    def __init__(self, inp_dim, nodes, nstack=4, tail_bound=4., tails='linear', num_bins=10, activation=F.relu,
                 **kwargs):
        super(StackedNonInvertibleNSF, self).__init__()
        transform_list = []
        for i in range(nstack):
            # If a tail function is passed apply the same tail bound to every layer, if not then only use the tail bound on
            # the final layer
            tpass = tails
            if tails:
                tb = tail_bound
            else:
                tb = tail_bound if i == 0 else None
            transform_list += [
                PureNonInvertibleNSF(inp_dim, nodes,
                                     num_blocks=2,
                                     tail_bound=tb, num_bins=num_bins,
                                     tails=tpass,
                                     activation=activation,
                                     use_residual_blocks=True)]
            if (tails == None) and (not tail_bound == None) and (i == nstack - 1):
                transform_list += [transforms.standard.PointwiseAffineTransform(-tail_bound, 2 * tail_bound)]

            transform_list += [transforms.ReversePermutation(inp_dim)]
        self.transform = transforms.CompositeTransform(transform_list[:-1])

    def forward(self, x, get_det=0, **kwargs):
        if get_det:
            return self.transform(x)
        else:
            return self.transform(x)[0]


class LULin(transforms.LULinear):
    def __init__(
            self,
            input_dim
    ):
        super(LULin, self).__init__(input_dim)

    def forward(self, inputs, **kwargs):
        return super().forward(inputs)[0]


class Flip(transforms.ReversePermutation):
    def __init__(
            self,
            input_dim
    ):
        super(Flip, self).__init__(input_dim)

    def forward(self, inputs, **kwargs):
        return super().forward(inputs)[0]
