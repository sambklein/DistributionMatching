from torch.nn.utils import clip_grad_norm_
from tqdm import trange
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

from dmatch.utils.io import get_top_dir, on_cluster

import time


class TimerError(Exception):
    """Timer exception."""


class Timer:
    def __init__(self, dir, nm, text="One cycle takes on average {:.1f} seconds"):
        self.text = text
        self._start_time = None
        self.cnt = 0
        self.times = []
        sv_dir = get_top_dir() + '/logs/' + dir
        if not os.path.exists(sv_dir):
            os.makedirs(sv_dir)
        self.file_name = sv_dir + '/' + nm

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self, sv=True):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        self.times += [elapsed_time]
        self.report()
        if sv:
            self.save_time()

    def save_time(self):
        self.report()
        with open(self.file_name, 'w') as f:
            f.write('{}\n'.format(self.mean_time))

    def report(self):
        self.mean_time = np.mean(self.times)
        if self.text:
            print(self.text.format(self.mean_time))


def get_vars(optim):
    return optim.param_groups[0]['params']


def get_grads(loss, vars):
    # This returns the gradients of loss wrt vars
    return torch.autograd.grad(loss, vars, retain_graph=True, allow_unused=True)
    # try:
    #     return torch.autograd.grad(loss, vars, retain_graph=True, allow_unused=True)
    # except:
    #     print('Gradient does not work')


def step_optimizers(grads, vars, optimizer, grad_norm_clip_value=None):
    # Set any gradients to zero before updating
    optimizer.zero_grad()
    for j, var in enumerate(vars):
        var.grad = grads[j]
    if grad_norm_clip_value is not None:
        clip_grad_norm_(vars, grad_norm_clip_value)
    optimizer.step()


def fit(model, optimizers, dataset, n_epochs, batch_size, writer, schedulers=None,
        regularizers=None, schedulers_epoch_end=None, gclip=None,
        monitor_interval=None, nsplit=5, plot_history=True, pin_memory=False):
    """
    :param model: the model to be trained
    :param optimizers: a list of optimizers corresponding to the sub models of the passed model
    :param dataset: data to train on, instance of torch.utils.dataset
    :param n_epochs: number of epochs to train the model for
    :param batch_size: the batch size used to make the torch.utils.data.DataLoader
    :param writer: a SummaryWriter from tensorboardX
    :param schedulers: list of learning rate schedulers for the corresponding optimizers
    :param regularizers: list of functions that return regularizers for the params in the corresponding optimizers list
    :param monitor_interval: number of steps at which to print the training statistics
    :return: A model trained according to the defined procedure
    """

    top_dir = get_top_dir()
    sv_dir = top_dir + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    # Initialize timer class, this is useful on the cluster as it will say if you have enough time to run the job
    timer = Timer(model.dir, '/timing_{}.txt'.format(model.exp_name))

    # ndata = dataset.data.shape[0]
    kfold = KFold(n_splits=nsplit, shuffle=True)
    inds = kfold.split(dataset)
    # TODO: implement the full cross validation process if doing hyper parameter scans, will need to handle val_loss saving
    # TODO: check the use of this writer object that you are passing - can't remeber what it is for :(
    train_inds, val_inds = list(inds)[0]
    # TODO: don't do this....
    try:
        trainset = dataset[train_inds]
        valset = dataset[val_inds]
        ntrain = trainset.data.shape[0]
        trainset.data = trainset.data[:ntrain - (ntrain % batch_size), :]
    except ValueError:
        trainset, valset = torch.utils.data.random_split(dataset, [50000, 10000])

    n_work = 2 if on_cluster() else 0
    training_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_work,
                                                pin_memory=pin_memory)
    val_data = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=n_work,
                                           pin_memory=pin_memory)

    # If a monitoring interval is not passed set a default
    if not monitor_interval:
        monitor_interval = 100

    tbar = trange(n_epochs, position=0, leave=True)
    trec = trange(n_epochs, desc='{desc}', position=2)
    tval = trange(n_epochs, desc='{desc}', position=3)
    # Get the names of the model losses
    loss_nms = model.loss_names
    # Make a pandas dataframe to store validation losses and one for train losses
    train_save = pd.DataFrame(columns=loss_nms)
    val_save = pd.DataFrame(columns=loss_nms)
    global_step = 0
    for epoch in tbar:
        timer.start()

        global_step += 1
        # TODO: there should also be exceptions for not passing the same number of regularizers per optimizer and schedule etc
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if regularizers:
            if not isinstance(regularizers, list):
                regularizers = [regularizers]

        if schedulers:
            # Don't define a scheduler if None is passed!
            if not isinstance(schedulers, list):
                schedulers = [schedulers]

        running_loss = []
        for i, data in enumerate(training_data, 0):
            if isinstance(data, list):
                data = data[0]
            data = data.to(model.device)

            # # zero the parameter gradients before calculating the losses
            [optimizer.zero_grad() for optimizer in optimizers]

            losses = model.compute_loss(data, batch_size)
            if not isinstance(losses, list):
                losses = [losses]

            if regularizers:
                regs = [reg() for reg in regularizers]
                # Add the regularization terms to the corresponding losses
                losses = [sum(x) for x in zip(losses, regs)]

            vars = [get_vars(optimizer) for optimizer in optimizers]
            grads = [get_grads(*pairs) for pairs in zip(losses, vars)]
            [step_optimizers(*triplets, grad_norm_clip_value=gclip) for triplets in zip(grads, vars, optimizers)]

            if schedulers:
                [scheduler.step() for scheduler in schedulers]

            if i % monitor_interval == 0:
                summaries = model.get_loss_state(4)
                running_loss += [list(summaries.values())]

                s = '[{}, {}] {}'.format(epoch + 1, i + 1, summaries)
                if not on_cluster():
                    trec.set_description_str(s)

                for summary, value in summaries.items():
                    writer.add_scalar(tag=summary, scalar_value=value, global_step=global_step)

        # save the mean training loss for the epoch
        train_loss = np.mean(np.array(running_loss), axis=0)
        train_info = dict(zip(loss_nms, train_loss))
        train_save = train_save.append(train_info, ignore_index=True)

        # Save the latest model, overwriting the last save
        path = top_dir + '/experiments/data/saved_models/model_{}'.format(model.exp_name)
        model.save(path)

        # TODO: get this to work correctly and nicely
        val_loss = []
        for i, data in enumerate(val_data, 0):
            if isinstance(data, list):
                data = data[0]
            data = data.to(model.device)
            # This updates the internal loss states which we want to access, the actual outputs are weighted sums
            model.compute_loss(data, batch_size)
            # Get the current loss state to 4 sf
            val_loss += [list(model.get_loss_state(4).values())]
        # Calculate the averages and make a dictionary of the losses.
        # TODO in cross-val we are interested in the full list - pandas dataframe
        val_loss = np.mean(np.array(val_loss), axis=0)
        val_info = dict(zip(loss_nms, val_loss))
        s = 'Validation Losses: ' + str(val_info)
        if not on_cluster():
            tval.set_description_str(s)
        val_save = val_save.append(val_info, ignore_index=True)

        # If the best validation error doesn't exist create it - useful for reloading models
        loss_track = np.mean(list(val_info.values()))
        try:
            best_val
        except NameError:
            best_val = loss_track
        # Save the best model
        if loss_track <= best_val:
            best_val = loss_track
            path = top_dir + '/experiments/data/saved_models/model_{}_best'.format(model.exp_name)
            model.save(path)

        # Step schedulers that use the validation data for stepping
        if schedulers_epoch_end:
            [scheduler.step(np.mean(val_loss)) for scheduler in schedulers_epoch_end if scheduler]

        # Stop the timer
        timer.stop()

    if plot_history:
        fig, axs = plt.subplots(1, len(loss_nms), figsize=(20, 5))
        for j, ax in enumerate(fig.axes):
            nm = loss_nms[j]
            ax.plot(val_save[nm], label='test')
            ax.plot(train_save[nm], '--', label='validation')
            ax.set_title(nm)
            ax.legend()
            ax.set_ylabel("loss")
            ax.set_xlabel("epoch")
        fig.savefig(sv_dir + '/training_{}.png'.format(model.exp_name))

    print('\nFinished Training')


# TODO: at present this makes for a lot of code duplication, don't delete the other method, but combine some of the functionality
def fit_generator(model, optimizers, generator, n_epochs, batch_size, writer, schedulers=None,
                  regularizers=None, schedulers_epoch_end=None, gclip=None,
                  monitor_interval=None, nsplit=5, plot_history=True):
    """
    :param model: the model to be trained
    :param optimizers: a list of optimizers corresponding to the sub models of the passed model
    :param dataset: data to train on, instance of torch.utils.dataset
    :param n_epochs: number of epochs to train the model for
    :param batch_size: the batch size used to make the torch.utils.data.DataLoader
    :param writer: a SummaryWriter from tensorboardX
    :param schedulers: list of learning rate schedulers for the corresponding optimizers
    :param regularizers: list of functions that return regularizers for the params in the corresponding optimizers list
    :param monitor_interval: number of steps at which to print the training statistics
    :return: A model trained according to the defined procedure
    """

    top_dir = get_top_dir()
    sv_dir = top_dir + '/images' + '/' + model.dir
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    # If a monitoring interval is not passed set a default
    if not monitor_interval:
        monitor_interval = 100

    tbar = trange(n_epochs, position=0, leave=True)
    trec = trange(n_epochs, desc='{desc}', position=2)
    tval = trange(n_epochs, desc='{desc}', position=3)
    # Get the names of the model losses
    loss_nms = model.loss_names
    # Make a pandas dataframe to store validation losses and one for train losses
    train_save = pd.DataFrame(columns=loss_nms)
    val_save = pd.DataFrame(columns=loss_nms)

    # Initialize timer class, this is useful on the cluster as it will say if you have enough time to run the job
    timer = Timer(model.dir, '/timing_{}.txt'.format(model.exp_name))

    global_step = 0
    for epoch in tbar:
        timer.start()
        global_step += 1
        # TODO: there should also be exceptions for not passing the same number of regularizers per optimizer and schedule etc
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if regularizers:
            if not isinstance(regularizers, list):
                regularizers = [regularizers]

        if schedulers:
            # Don't define a scheduler if None is passed!
            if not isinstance(schedulers, list):
                schedulers = [schedulers]

        running_loss = []
        for i in range(generator.nsteps):
            data = generator.get_data(i)
            if isinstance(data, list):
                data = data[0]
            data = data.to(model.device)

            # # zero the parameter gradients before calculating the losses
            [optimizer.zero_grad() for optimizer in optimizers]

            losses = model.compute_loss(data, batch_size)
            if not isinstance(losses, list):
                losses = [losses]

            if regularizers:
                regs = [reg() for reg in regularizers]
                # Add the regularization terms to the corresponding losses
                losses = [sum(x) for x in zip(losses, regs)]

            vars = [get_vars(optimizer) for optimizer in optimizers]
            grads = [get_grads(*pairs) for pairs in zip(losses, vars)]
            [step_optimizers(*triplets, grad_norm_clip_value=gclip) for triplets in zip(grads, vars, optimizers)]

            if schedulers:
                [scheduler.step() for scheduler in schedulers]

            if i % monitor_interval == 0:
                summaries = model.get_loss_state(4)
                running_loss += [list(summaries.values())]

                s = '[{}, {}] {}'.format(epoch + 1, i + 1, summaries)
                if not on_cluster():
                    trec.set_description_str(s)

                for summary, value in summaries.items():
                    writer.add_scalar(tag=summary, scalar_value=value, global_step=global_step)

        # save the mean training loss for the epoch
        train_loss = np.mean(np.array(running_loss), axis=0)
        train_info = dict(zip(loss_nms, train_loss))
        train_save = train_save.append(train_info, ignore_index=True)

        # Save the latest model, overwriting the last save
        path = top_dir + '/experiments/data/saved_models/model_{}'.format(model.exp_name)
        print(f'Model saved at {path}')
        model.save(path)

        # TODO: get this to work correctly and nicely
        val_loss = []
        for i in range(generator.nsteps_val):
            data = generator.get_val_data(i)
            if isinstance(data, list):
                data = data[0]
            data = data.to(model.device)
            # This updates the internal loss states which we want to access, the actual outputs are weighted sums
            model.compute_loss(data, batch_size)
            # Get the current loss state to 4 sf
            val_loss += [list(model.get_loss_state(4).values())]
        # Calculate the averages and make a dictionary of the losses.
        # TODO in cross-val we are interested in the full list - pandas dataframe
        val_loss = np.mean(np.array(val_loss), axis=0)
        val_info = dict(zip(loss_nms, val_loss))
        s = 'Validation Losses: ' + str(val_info)
        if not on_cluster():
            tval.set_description_str(s)
        val_save = val_save.append(val_info, ignore_index=True)

        # If the best validation error doesn't exist create it - useful for reloading models
        loss_track = np.mean(list(val_info.values()))
        try:
            best_val
        except NameError:
            best_val = loss_track
        # Save the best model
        if loss_track <= best_val:
            best_val = loss_track
            path = top_dir + '/experiments/data/saved_models/model_{}_best'.format(model.exp_name)
            model.save(path)

        # Step schedulers that use the validation data for stepping
        if schedulers_epoch_end:
            [scheduler.step(np.mean(val_loss)) for scheduler in schedulers_epoch_end if scheduler]

        timer.stop()

    if plot_history:
        fig, axs = plt.subplots(1, len(loss_nms), figsize=(20, 5))
        for j, ax in enumerate(fig.axes):
            nm = loss_nms[j]
            ax.plot(val_save[nm], label='test')
            ax.plot(train_save[nm], '--', label='validation')
            ax.set_title(nm)
            ax.legend()
            ax.set_ylabel("loss")
            ax.set_xlabel("epoch")
        fig.savefig(sv_dir + '/training_{}.png'.format(model.exp_name))

    print('\nFinished Training')


def fit_classifier(classifier, train_data, valid_data, optimizer, batch_size, n_epochs, device, sv_dir, plot=True,
                   save=True):
    # Make an object to load training data
    data_obj = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    data_valid = torch.utils.data.DataLoader(valid_data, batch_size=1000, shuffle=True, num_workers=0)
    n_train = int(np.ceil(len(train_data) / batch_size))
    n_valid = int(np.ceil(len(valid_data) / 1000))

    train_loss = np.zeros(n_epochs)
    valid_loss = np.zeros(n_epochs)
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = np.zeros(n_train)
        for i, data in enumerate(data_obj, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # Get the model loss
            data = [dt.to(device) for dt in data]
            loss = classifier.compute_loss(data)
            # Propogate the loss
            loss.backward()
            # Update the parameters
            optimizer.step()

            # Get statistics
            running_loss[i] = loss.item()

        # Save loss info for the epoch
        train_loss[epoch] = np.mean(running_loss)

        # Validate
        running_loss = np.zeros(n_valid)
        with torch.no_grad():
            for i, data in enumerate(data_valid, 0):
                # Get the model loss
                loss = classifier.compute_loss(data)
                running_loss[i] = loss.item()
        valid_loss[epoch] = np.mean(running_loss)

    if plot:
        plt.figure()
        plt.plot(train_loss, label='Train')
        plt.plot(valid_loss, label='Validation')
        plt.legend()
        plt.title('Classifier Training')
        plt.tight_layout()
        plt.savefig(sv_dir + 'Training.png')

    if save:
        classifier.save(sv_dir + 'classifier')

    print('Finished Training')


def get_log_det(data, model):
    """This works by implementing a set of linear equations that can be solved for the Jacobian."""
    if data.shape[1] != 2:
        raise Exception('This implementation only works on two dimensional data.')
    x = data
    x = x.squeeze()
    batch_size = int(x.shape[0])
    tk2 = torch.ones_like(x)
    tk2[:, 1] = -1
    grad_tensor = torch.cat((torch.ones_like(x), tk2))
    x = x.repeat(2, 1)
    x.requires_grad_(True)
    y = model(x)
    # y.backward(grad_tensor, create_graph=True)
    # jacobian_solutions = torch.cat((x.grad[:batch_size], x.grad[batch_size:]), 1)
    grads = torch.autograd.grad(y, x, grad_outputs=grad_tensor, create_graph=True)[0]
    jacobian_solutions = torch.cat((grads[:batch_size], grads[batch_size:]), 1)

    def invert(solutions):
        b = (solutions[:, 0] - solutions[:, 1]) / 2
        a = solutions[:, 0] - b
        return a, b

    c1, c3 = invert(jacobian_solutions[:, ::2])
    c2, c4 = invert(jacobian_solutions[:, 1::2])
    abs_log_detJ = (c1 * c4 - c2 * c3).abs().log()
    abs_log_detJ[abs_log_detJ.isinf()] = 0
    return abs_log_detJ

    # # Can test with the below
    # for j in range(data.shape[0]):
    #     det_j_auto = det_j_auto + torch.autograd.functional.jacobian(model,
    #                                                                  data[j].view(-1, 2),
    #                                                                  create_graph=True).squeeze().det().abs().log()