import torch

from dmatch.data.plane import GaussianDataset, CheckerboardModes
from dmatch.data.plane import CrescentDataset
from dmatch.data.plane import CrescentCubedDataset
from dmatch.data.plane import SineWaveDataset
from dmatch.data.plane import AbsDataset
from dmatch.data.plane import SignDataset
from dmatch.data.plane import FourCircles
from dmatch.data.plane import DiamondDataset
from dmatch.data.plane import TwoSpiralsDataset
from dmatch.data.plane import CheckerboardDataset
from dmatch.data.plane import CornersDataset
from dmatch.data.plane import EightGaussiansDataset

from dmatch.data.hyper_dim import HyperCheckerboardDataset, SparseHyperCheckerboardDataset, HyperShells, \
    HyperSouthernCross

# from jets_utils.jutils.data.data_loaders import load_jets as jutils_load_jets

import os
import pandas as pd
import numpy as np
import h5py

# Taken from https://github.com/bayesiains/nsf/blob/master/data/base.py
from dmatch.utils.io import get_top_dir, on_cluster


def load_plane_dataset(name, num_points, flip_axes=False, scale=True, npad=0, dim=None):
    """Loads and returns a plane dataset.
    Args:
        name: string, the name of the dataset.
        num_points: int, the number of points the dataset should have,
        flip_axes: bool, flip x and y axes if True.
    Returns:
        A Dataset object, the requested dataset.
    Raises:
         ValueError: If `name` an unknown dataset.
    """

    try:
        if dim:
            dataset = {
                'checkerboard': HyperCheckerboardDataset,
                'sparsecheckerboard': SparseHyperCheckerboardDataset,
                'stars': HyperSouthernCross,
                'shells': HyperShells

            }[name](num_points=num_points, dim=dim, flip_axes=flip_axes)
        else:
            dataset = {
                'gaussian': GaussianDataset,
                'crescent': CrescentDataset,
                'crescent_cubed': CrescentCubedDataset,
                'sine_wave': SineWaveDataset,
                'abs': AbsDataset,
                'sign': SignDataset,
                'four_circles': FourCircles,
                'diamond': DiamondDataset,
                'two_spirals': TwoSpiralsDataset,
                'checkerboard': CheckerboardDataset,
                'corners': CornersDataset,
                'eightgauss': EightGaussiansDataset,
                'hypercheckerboard': HyperCheckerboardDataset,
                'checkerboard_modes': CheckerboardModes,

            }[name](num_points=num_points, flip_axes=flip_axes)
        if scale:
            # Scale data to be between zero and one
            # dataset.data = 2 * (dataset.data - dataset.data.min()) / (dataset.data.max() - dataset.data.min()) - 1
            dataset.data = (dataset.data + 4) / 4 - 1
        if npad > 0:
            padder = torch.distributions.uniform.Uniform(torch.zeros(npad), torch.ones(npad), validate_args=None)
            pads = padder.sample([num_points])
            dataset.data = torch.cat((dataset.data, pads), 1)
        return dataset

    except KeyError:
        raise ValueError('Unknown dataset: {}'.format(name))


# TODO: this is stupid code duplication from the pytorch-utils repo in the anomaly tools.
def fill_array(to_fill, obj, dtype):
    arr = np.array(obj, dtype=dtype)
    to_fill[:len(arr)] = arr


# A class for generating data for plane datasets.
class data_handler():
    def __init__(self, nsample, batch_size, latent_dim, dataset, device):
        self.nsample = nsample
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.dataset = dataset
        self.device = device
        self.dim = None if not dataset[:5] == 'hyper' else self.latent_dim
        self.bounded = load_plane_dataset(self.dataset, 8, dim=self.dim).bounded
        self.scale = 1.
        self.update_data()
        self.nsteps = int(self.nsample / self.batch_size)
        # TODO: pass a steps valid parameter to define this properly
        self.nval = int(self.nsample / 10)
        self.nsteps_val = int(self.nval / self.batch_size)

    def update_data(self):
        trainset = load_plane_dataset(self.dataset, self.nsample, dim=self.dim)
        self.data = trainset.data.to(self.device).view(-1, self.batch_size, self.latent_dim) * self.scale

    def sample(self, n_sample):
        if isinstance(n_sample, list):
            n_sample = n_sample[0]
        return load_plane_dataset(self.dataset, n_sample, dim=self.dim).data * self.scale

    def update_validation(self):
        trainset = load_plane_dataset(self.dataset, int(self.nsample / 10), dim=self.dim)
        self.valid = trainset.data.to(self.device).view(-1, self.batch_size, self.latent_dim) * self.scale

    def get_data(self, i):
        # On the start of each epoch generate new samples, and then for each proceeding epoch iterate through the data
        if i == 0:
            self.update_data()
        return self.data[i]

    def get_val_data(self, i):
        # On the start of each epoch generate new samples, and then for each proceeding epoch iterate through the data
        if i == 0:
            self.update_validation()
        return self.valid[i]

# def main():
#     import argparse
#     import matplotlib.pyplot as plt
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--download', type=int, default=0,
#                         help='Choose the base output directory')
#
#     args = parser.parse_args()
#
#     if args.download:
#         load_hepmass(mass='1000')
#         load_hepmass(mass='all')
#         load_hepmass(mass='not1000')
#
#     data_train, data_test = load_hepmass(mass='1000', slim=True)
#
#     fig, axs_ = plt.subplots(9, 3, figsize=(5 * 3 + 2, 5 * 9 + 2))
#     axs = fig.axes
#     for i, data in enumerate(data_train.data.t()):
#         axs[i].hist(data.numpy())
#     fig.savefig(get_top_dir() + '/images/hepmass_features.png')
#
#     return 0


# if __name__ == '__main__':
#     main()
