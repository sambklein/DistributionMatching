import matplotlib.pyplot as plt
import numpy as np
import torch

import h5py

from ..data.hyper_dim import HyperCheckerboardDataset

from .io import get_top_dir, on_cluster
from .plotting import plot2Dhist, getSummaryplot, plot_latent_embeddings, plot_samples_mnist, getCrossFeaturePlot, \
    plot_auc, plot_salience

import os

# TODO: the argument you have called vae should be a stochastic argument, so you can see encoded means etc.
from .torch_utils import batch_function


def post_process_plane(model, test_loader, invertible=False, implicit=True, sup_title='', bounds=[-1.5, 1.5], bins=50,
                       vae=False):
    test_loader = test_loader.data.to(model.device)
    model.eval()
    nm = model.exp_name
    top_dir = get_top_dir()
    sv_dir = top_dir + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    with torch.no_grad():
        # Get all outputs from forward pass of the model
        # outputs = model.get_outputs(test_loader.data, int(1e4), internal=True)
        outputs = model.get_outputs(test_loader.data, int(1e4))
        # Use if statements to calculate the number of plots that will be created TODO: this is quite a silly way to do this
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            nplots = sum([output.shape[1] <= 2 for output in outputs])
            # if outputs[0].shape[1] <= 2:
            #     nplots = len(outputs)
        else:
            nplots = 1
        # There is one plot if it is and if it isn't invertible
        nplots += 2
        if implicit:
            nplots += 1

        fig, axs_ = plt.subplots(1, nplots, figsize=(5 * nplots + 2, 5))
        axs = fig.axes

        # Plot samples drawn from the model
        samples = model.get_numpy(model.sample(int(1e5), int(1e3)))
        plot2Dhist(samples, axs[0], bins, bounds)
        axs[0].set_title('Samples')
        # Create an index for creating the other plots
        ind = 1

        # If the inner model is not invertible show the outer encoder performance
        if not invertible:
            # recons = model.get_numpy(model.autoencode(test_loader.to(device)))
            encoding = batch_function(model.encoder, test_loader, int(1e4))
            # If this is a VAE there are two outputs from the encoder
            if vae:
                encoding = model.model_sample(encoding)
            recons = model.get_numpy(batch_function(model.decoder, encoding, int(1e4)))
            plot2Dhist(recons, axs[ind], bins, bounds)
            axs[ind].set_title('Reconstruction')
            ind += 1

        if invertible:
            encoding = model.get_numpy(model.encode(test_loader))
            plot2Dhist(encoding, axs[ind], bins)
            axs[ind].set_title('Encoding')
            ind += 1

        # Visualize the different latent spaces
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            latent_names = ['A', 'Z', "A'"]
            for i, l_spc in enumerate(outputs[:-1]):
                l_spc = model.get_numpy(l_spc)
                if l_spc.shape[1] == 2:
                    plot2Dhist(l_spc, axs[ind], bins)
                    axs[ind].set_title('Latent Space {}'.format(latent_names[i]))
                    ind += 1
                if l_spc.shape[1] == 1:
                    axs[ind].hist(l_spc, bins)
                    axs[ind].set_title('Latent Space {}'.format(latent_names[i]))
                    ind += 1

        if implicit:
            # plot samples in A space
            a_sample = batch_function(model.zy, model.bdist_sample(int(1e5)), int(1e4))
            if isinstance(a_sample, tuple):
                a_sample = model.model_sample(a_sample)
                # a_sample = a_sample[0]
            a_sample = model.get_numpy(a_sample)

            plot2Dhist(a_sample, axs[ind], bins)
            axs[ind].set_title('A space sample')

        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fig.suptitle(sup_title + ' {}'.format(nparams))
        fig.tight_layout()
        fig.savefig(sv_dir + '/post_processing_{}.png'.format(nm))

        print('There are {} trainable parameters'.format(nparams))

        if vae:
            # Plot the means and std
            encoding = model.encoder(test_loader)
            fig, axs_ = plt.subplots(1, 2, figsize=(5 * nplots + 2, 5))
            axs = fig.axes
            titles = ['Means', 'Logvars']
            for i in range(2):
                plot2Dhist(model.get_numpy(encoding[i]), axs[i], bins)
                axs[i].set_title(titles[i])
            fig.tight_layout()
            fig.savefig(sv_dir + '/post_processing_means_{}.png'.format(nm))

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        recons = model.get_numpy(batch_function(model.autoencode, test_loader, int(1e4)))
        plot2Dhist(recons, axs, bins)
        fig.tight_layout()
        fig.savefig(sv_dir + '/autoencode_{}.png'.format(nm))

        # TODO: Sinkhorn distace of samples from dataset
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plot2Dhist(model.get_numpy(encoding), ax, 100)
        # ax.set_title(' {}'.format(nparams))
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                       labelleft=False)
        lim = 3.5
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        # plt.xticks([-3, 0, 3])
        # plt.yticks([-3, 0, 3])
        fig.tight_layout()
        fig.savefig(sv_dir + '/encoded_distribution_{}.png'.format(nm))


def get_counts(data, to_slice, bound=4, nbins=50):
    bin_edges = np.linspace(-bound, bound, nbins + 1)
    # Apply a slice to the data
    mask = torch.all((to_slice > 0) & (to_slice < 2), 1)
    data = data[mask.type(torch.bool)].cpu().numpy()
    return np.histogram2d(data[:, 0], data[:, 1], bins=bin_edges)[0]


def get_ood(model, nsamples, nrun, bound, nbins, data_generator=None, get_target=None, max_it=1000):
    percent_ood = np.zeros(nrun)
    percent_oob = np.zeros(nrun)
    counts = np.zeros((nbins, nbins))
    counts_true = np.zeros((nbins, nbins))
    it = 0

    for i in range(nrun):
        with torch.no_grad():
            # TODO: deal with this code duplication
            if data_generator:
                data = data_generator(nsamples)
                sample = model.batch_predict(data).detach().cpu()
            else:
                # If not generator is passed then the model must be a flow
                sample = model.sample(nsamples).detach().cpu()
        percent_ood[i] = HyperCheckerboardDataset.count_ood(sample)
        percent_oob[i] = HyperCheckerboardDataset.count_oob(sample)
        counts += get_counts(sample, sample[:, 2:], bound, nbins)
        if get_target:
            target = get_target(data.view(-1, data.shape[-1]))
            counts_true += get_counts(target, target[:, 2:], bound, nbins)

    print('{}% of OOD data, std {}.'.format(np.mean(percent_ood), np.std(percent_ood)))
    print('{}% of OOB data, std {}.'.format(np.mean(percent_oob), np.std(percent_oob)))

    # Plot one slice of the data to inspect the training.
    # TODO: shift any improvements back to hyper_dim.py testing, and ideally load this as a function
    # TODO: should also update percent_ood in the while loop - but it seems to be quite accurate

    while np.sum(counts) < int(1e4) and (it < max_it):
        it += 1
        with torch.no_grad():
            if data_generator:
                data = data_generator(nsamples)
                sample = model.batch_predict(data).detach().cpu()
            else:
                # If not generator is passed then the model must be a flow
                sample = model.sample(nsamples).detach().cpu()
        counts += get_counts(sample, sample[:, 2:], bound, nbins)
        if get_target:
            target = get_target(data)
            counts_true += get_counts(target, target[:, 2:], bound, nbins)
    if get_target:
        return percent_ood, percent_oob, counts, counts_true
    else:
        return percent_ood, percent_oob, counts


def get_salience(dist, name, model, nsamp, batch_size, target_dist, device, dist_measure, sv_dir, exp_name,
                 additional_loss=None, beta=0.):
    n_run = int(nsamp // batch_size)
    latent_dim = dist.shape[1]
    uniform_sample = dist.view(n_run, batch_size, latent_dim)
    output_sample = torch.zeros_like(uniform_sample)
    saliency = torch.zeros((n_run, batch_size))

    for i in range(n_run):
        x = uniform_sample[i]
        x.requires_grad_()
        if x.grad is not None:
            x.grad.zero_()
        # Get the model output
        encoding = model(x)
        sample = target_dist.sample([batch_size]).to(device)
        if additional_loss is not None:
            add_loss = additional_loss(x)

        if (additional_loss is not None) and (beta > 0.):
            scores = dist_measure(encoding, sample) + beta * add_loss
        elif (additional_loss is not None) and (beta == 0.):
            scores = add_loss
        else:
            scores = dist_measure(encoding, sample)

        scores.backward(torch.ones_like(scores))
        saliency[i] = x.grad.norm(dim=1).cpu()
        output_sample[i] = encoding.cpu()

    saliency = saliency.view(-1, 1)
    x = uniform_sample.view(-1, latent_dim)
    y = output_sample.view(-1, latent_dim)
    plot_salience(x, y, saliency[:, 0], sv_dir + f'/encoded_saliency{name}_{exp_name}.png')
    return x, y, saliency


def get_samples_model(transformer, base_dist, model_name, data_name, device, scale_fact=1):
    sv_dir = get_top_dir() + '/dmatch/data/slims/models_for_paper'
    nsamples = int(5e6)
    batch_size = int(1e5)
    n_batches = int(nsamples / batch_size)
    samples = np.zeros((n_batches, batch_size, 2))
    with torch.no_grad():
        for i in range(n_batches):
            samples[i] = transformer(base_dist.sample([batch_size]).to(device) * scale_fact).cpu().numpy()
    samples = samples.reshape(-1, 2)
    with h5py.File(sv_dir + f'/{model_name}_{data_name}.h5', 'w') as hf:
        hf.create_dataset('samples', data=samples)
