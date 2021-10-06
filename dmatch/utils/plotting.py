# Some plotting functions
import colorsys
import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt, colors as colors
from sklearn.manifold import TSNE
import seaborn as sns
from dmatch.utils.torch_utils import get_numpy


def plot_density(data, nm, bottom=True, left=True, right=False, ticks=[-3, 0, 3], lim=4, ax=None, clip=False,
                 pad_y_ticks=30, label_fonts=18):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = None

    if lim is not None:
        bnd = [-lim, lim]
    else:
        bnd = None

    plot2Dhist(data, ax, 100, bounds=bnd, clip_val=clip)

    if right:
        ax.yaxis.tick_right()
    if not bottom:
        # ax.tick_params(axis='x', which='both', labelbottom=False)
        [t.set_color('white') for t in ax.xaxis.get_ticklabels()]
    if not left:
        # ax.tick_params(axis='y', which='both', labelleft=False)
        [t.set_color('white') for t in ax.yaxis.get_ticklabels()]
    # ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
    #                labelleft=False)

    if lim is None:
        lim = max(ax.get_ylim() + ax.get_xlim())

    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_box_aspect(1)

    plt.xticks(fontsize=label_fonts)
    plt.yticks(fontsize=label_fonts, ha='right')
    ax.yaxis.set_tick_params(pad=pad_y_ticks)
    ax.tick_params(axis='both', width=2, length=10)

    if fig is not None:
        fig.tight_layout(pad=0.1)
        fig.savefig(nm)


def get_mask(x, bound):
    return np.logical_and(x > bound[0], x < bound[1])


def apply_bound(data, bound):
    mask = np.logical_and(get_mask(data[:, 0], bound), get_mask(data[:, 1], bound))
    return data[mask, 0], data[mask, 1]


def plot2Dhist(data, ax, bins=50, bounds=None, clip_val=False):
    if bounds:
        x, y = apply_bound(data, bounds)
    else:
        x = data[:, 0]
        y = data[:, 1]
    count, xbins, ybins = np.histogram2d(x, y, bins=bins)
    count[count == 0] = np.nan
    if clip_val:
        clip_val = np.nanquantile(count, 0.01)
    ax.imshow(count.T,
              origin='lower', aspect='auto',
              extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
              vmin=clip_val
              )


def get_colors(num_colors):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def plot_embedding(embed, y_test, clrs, ax=None, title='', dims=(8, 8)):
    if not ax:
        plt.figure(figsize=dims)
        ax = plt
    else:
        ax.set_title(title)
    for i in range(9):
        mx = y_test == i
        ax.plot(embed[:, 0][mx], embed[:, 1][mx], 'x', color=clrs[i], alpha=0.3)


def hist_latents(inp, title='', bins=20):
    fig, ax = plt.subplots(1, inp.shape[1], figsize=(20, 5))
    fig.suptitle(title, fontsize=16)
    for i in range(inp.shape[1]):
        ax[i].hist(inp[:, i], bins=bins)


def plot_latent_embeddings(enc_ims, labels, title=None):
    # enc_ims = encoder(data)
    if enc_ims.shape[1] > 2:
        x_embeddor = TSNE(n_components=2)
        X_emb = x_embeddor.fit_transform(enc_ims)
    else:
        X_emb = enc_ims

    clrs = get_colors(9)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_embedding(X_emb, labels, clrs, ax=ax, title='Latent Space')
    if title:
        plt.savefig(title)


def plot_samples_mnist(samples, ax=None):
    samples = np.array(samples)

    for i, sample in enumerate(samples):
        sample = sample.reshape(28, 28)
        ax[i].imshow(sample)


def plot_slice(counts, nm, bound=4):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    counts[counts == 0] = np.nan
    ax.imshow(counts.T,
              origin='lower', aspect='auto',
              extent=[-bound, bound, -bound, bound],
              )
    fig.savefig(nm)


def plot_coloured(data, to_mask, ax, name, base_dist, alpha=0.1, set_limits=True, filler=None, filler_alpha=0.05):
    try:
        to_mask = to_mask.detach().cpu().numpy()
        filler = filler.detach().cpu().numpy()
    except Exception as e:
        print(e)
        pass

    marker_size = 2
    x = data[:, 0]
    y = data[:, 1]
    mx_low_prob = (x ** 2 + y ** 2) ** (0.5) > 3
    mx_checkers = ((y / 2).floor() % 2).type(torch.bool)
    not_checkerboard = base_dist[:12] != 'checkerboard'

    # if filler is not None:
    #     ax.scatter(filler[:, 0], filler[:, 1], s=marker_size, color='black', alpha=filler_alpha)

    def scatter(mx, color, alpha):
        # This must also be cast to a numpy, otherwise the masked array is also a tensor
        mx = mx.detach().cpu().numpy()
        ax.scatter(to_mask[mx, 0], to_mask[mx, 1], s=marker_size, color=color, alpha=alpha)

    def scatter_mask(id, color1, color2):
        if not_checkerboard:
            mx = id & ~mx_low_prob
        else:
            mx = id & ~mx_checkers
        scatter(mx, color1, alpha)
        if not_checkerboard:
            mx = id & mx_low_prob
            scatter(mx, color2, alpha)
        else:
            mx = id & mx_checkers
            scatter(mx, color2, alpha)

    id = torch.logical_and(x < 0, y < 0)
    scatter_mask(id, 'darkred', 'indianred')

    id = torch.logical_and(x > 0, y < 0)
    scatter_mask(id, 'darkgoldenrod', 'burlywood')

    id = torch.logical_and(x < 0, y > 0)
    scatter_mask(id, 'darkblue', 'cornflowerblue')

    id = torch.logical_and(x > 0, y > 0)
    scatter_mask(id, 'darkslategrey', 'slategrey')

    bound = 4.5
    if set_limits:
        ax.set_xlim([-bound, bound])
        ax.set_ylim([-bound, bound])
    ax.set_title(name)


# From Johnny
def projectiontionLS_2D(dim1, dim2, latent_space, *args, **kwargs):
    '''Plot a two dimension latent space projection with marginals showing each dimension.
    Can overlay multiple different datasets by passing more than one latent_space argument.
    Inputs:
        dim1: First LS dimension to plot on x axis
        dim2: Second LS dimension to plot on y axis
        latent_space (latent_space2, latent_space3...): the data to plot
    Optional:
        xrange: specify xrange in form [xmin,xmax]
        yrange: specify xrange in form [ymin,ymax]
        labels: labels in form ['ls1','ls2','ls3'] to put in legend
        Additional options will be passed to the JointGrid __init__ function
    Returns:
        seaborn JointGrid object
    '''
    if 'xrange' in kwargs:
        xrange = kwargs.get('xrange')
    else:
        xrange = (np.floor(np.quantile(latent_space[:, dim1], 0.02)), np.ceil(np.quantile(latent_space[:, dim1], 0.98)))
    if 'yrange' in kwargs:
        yrange = kwargs.get('yrange')
    else:
        yrange = (np.floor(np.quantile(latent_space[:, dim2], 0.02)), np.ceil(np.quantile(latent_space[:, dim2], 0.98)))
    labels = [None] * (1 + len(args))
    if 'labels' in kwargs:
        labels = kwargs.get('labels')
    kwargs.pop('xrange', None)
    kwargs.pop('yrange', None)
    kwargs.pop('labels', None)
    g = sns.JointGrid(latent_space[:, dim1], latent_space[:, dim2], xlim=xrange, ylim=yrange, **kwargs)
    # for label in [0,1]:
    sns.kdeplot(latent_space[:, dim1], ax=g.ax_marg_x, legend=False, shade=True, alpha=0.3, label=labels[0])
    sns.kdeplot(latent_space[:, dim2], ax=g.ax_marg_y, vertical=True, legend=False, shade=True, alpha=0.3,
                label=labels[0])
    sns.kdeplot(latent_space[:, dim1], latent_space[:, dim2], ax=g.ax_joint, shade=True, shade_lowest=False, bw=0.2,
                alpha=1, label=labels[0])
    i = 1
    for ls in args:
        sns.kdeplot(ls[:, dim1], ax=g.ax_marg_x, legend=False, shade=True, alpha=0.3, label=labels[i])
        sns.kdeplot(ls[:, dim2], ax=g.ax_marg_y, vertical=True, legend=False, shade=True, alpha=0.3, label=labels[i])
        sns.kdeplot(ls[:, dim1], ls[:, dim2], ax=g.ax_joint, shade=True, shade_lowest=False, bw=0.2, alpha=0.4,
                    label=labels[i])
        i += 1
    g.ax_joint.spines['right'].set_visible(True)
    g.ax_joint.spines['top'].set_visible(True)
    g.set_axis_labels('LS Dim. {}'.format(dim1), 'LS Dim. {}'.format(dim2))
    if labels[0] is not None:
        g.ax_joint.legend()
    return g


def getSummaryplot(scores, file_name):
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    from matplotlib.cm import get_cmap

    scorekey = [keys for keys in scores.keys()]
    markers = ['o', 'v', 'd', 'p']
    bglabel = ['50% bkg rej', '80% bkg rej', '90% bkg rej', '95% bkg rej']
    bgpointer = [Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='k', markersize=8) for i in range(4)]

    cmap = get_cmap('Set1')
    colour = []
    for item in np.arange(0, 0.68, 0.68 / 3):
        colour.append(cmap(item))
    metricpointer = [mpatches.Patch(color=colors) for colors in colour]
    metricpointer.append(Line2D([0], [0], color='w', markerfacecolor='k', markersize=8))
    handles = bgpointer + metricpointer
    labels = bglabel + scorekey

    fig, ax = plt.subplots()
    for score, color in zip(scorekey, colour):
        for i in range(4):
            ax.scatter(scores[score][1][i], scores[score][0][i], marker=markers[i], color=color)
        ax.plot(scores[score][1], scores[score][0], color=color, linestyle='--', alpha=0.5)
    ax.set_xlabel('Signal Efficiencies')
    ax.set_ylabel('Mass Sculpting (KS)')
    plt.legend(handles, labels, columnspacing=0.95, ncol=2, framealpha=0.0, frameon=False, handlelength=1.8,
               handletextpad=0.6)
    plt.tight_layout()
    plt.savefig(file_name)


def get_bins(data, nbins=20):
    max_ent = data.max().item()
    min_ent = data.min().item()
    return np.linspace(min_ent, max_ent, num=nbins)


def getCrossFeaturePlot(data, nm, savedir):
    nfeatures = data.shape[1]
    fig, axes = plt.subplots(nfeatures, nfeatures,
                             figsize=(np.clip(5 * nfeatures + 2, 5, 22), np.clip(5 * nfeatures - 1, 5, 20)))
    for i in range(nfeatures):
        for j in range(nfeatures):
            if i == j:
                axes[i, i].hist(get_numpy(data[:, i]))
            elif i < j:
                bini = get_bins(data[:, i])
                binj = get_bins(data[:, j])
                axes[i, j].hist2d(get_numpy(data[:, i]), get_numpy(data[:, j]), bins=[bini, binj],
                                  density=True, cmap='Reds')
    fig.tight_layout()
    plt.savefig(savedir + '/feature_correlations_{}_{}.png'.format(nm, 'transformed_data'))


def plot_auc(roc_auc, fpr, tpr, sv_name):
    # Plot a roc curve
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'{roc_auc}')
    fig.savefig(sv_name)


def plot_spline(spline, dtype, sv, device, norm=None, fig=None, ax=None):
    from torch.nn import functional as F
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    x = np.linspace(-4, 4, 1000)
    with torch.no_grad():
        if norm is None:
            y = spline(torch.tensor(x, dtype=dtype).view(-1, 1).to(device)).cpu().numpy()
        else:
            y = spline(torch.tensor(x, dtype=dtype).view(-1, 1).to(device), norm=False).cpu().numpy()
    ax.plot(x, y)
    try:
        boundaries = spline.bound_func(spline.untransed_boundaries, spline.upper_bound, spline.lower_bound)
        heights = spline(boundaries)
        plt.plot(boundaries.detach().cpu().numpy(), heights.detach().cpu().numpy(), 'x')
    except:
        pass
    fig.savefig(sv)


def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def plot_salience(x, y, saliency, name, n_bins=200):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    saliency = saliency.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, array in list(zip(axes, [x, y])):
        means, bins_x, bins_y, _ = scipy.stats.binned_statistic_2d(array[:, 0], array[:, 1], saliency,
                                                                   'mean',
                                                                   bins=n_bins
                                                                   )
        # To use this as imshow you have to rotate
        means = np.rot90(means)
        extent = [array[:, 0].min(), array[:, 0].max(), array[:, 1].min(), array[:, 1].max()]
        im = ax.imshow(means, extent=extent, norm=colors.LogNorm())
        ticks = [-4, -2, 0, 2, 4]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        # add_colorbar(im)
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(name)
    plt.clf()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    clip_val = None
    for i, (ax, array) in enumerate(list(zip(axes, [x, y]))):
        means, bins_x, bins_y, _ = scipy.stats.binned_statistic_2d(array[:, 0], array[:, 1], saliency,
                                                                   'mean',
                                                                   bins=n_bins
                                                                   )
        # To use this as imshow you have to rotate
        means = np.rot90(means)
        extent = [array[:, 0].min(), array[:, 0].max(), array[:, 1].min(), array[:, 1].max()]
        if clip_val is None:
            clip_val = np.nanquantile(means, 0.9)
        im = ax.imshow(means, extent=extent, vmax=clip_val)
        ticks = [-4, -2, 0, 2, 4]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        # add_colorbar(im)
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(name.split('.')[0] + 'non_log.png')
    plt.clf()
