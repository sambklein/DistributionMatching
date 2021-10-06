import torch
import numpy as np
from dmatch.data.classsification_datasets import SupervisedDataClass
from dmatch.models.classifier import Classifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

import xgboost as xgb
from dmatch.models.nn.dense_nets import MLP
from dmatch.utils import fit_classifier


def get_auc_xgb(base_dist, target_dist, ndata, model=None, split=None):
    sample = split is None

    if sample:
        def get_data(half_size):
            normal_samples = base_dist.sample([half_size])
            if model is not None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                normal_samples = model(normal_samples.to(device)).detach().cpu()
            reference_samples = target_dist.sample([half_size])
            X = torch.cat((normal_samples, reference_samples)).numpy()
            Y = np.concatenate((np.zeros(half_size), np.ones(half_size)))
            return X, Y
        X_train, Y_train = get_data(ndata)
        X_valid, Y_valid = get_data(int(0.1 * ndata))
        X_test, Y_test = get_data(int(ndata))
    else:
        def get_data(data):
            n = len(data)
            return np.concatenate((data, target_dist.sample([n]).cpu())), np.concatenate((np.ones(n), np.zeros(n)))
        kf = KFold(n_splits=split)

    # print(X_train.shape, Y_train.shape)
    classifier = xgb.XGBClassifier(n_estimators=1000, max_depth=5, use_label_encoder=False, learning_rate=0.01,
                                   objective='binary:logistic', eta=0.1, gammea=0.1, verbosity=0)
    if sample:
        classifier.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)])
        y_scores = classifier.predict(X_test)
        fpr, tpr, _ = roc_curve(Y_test, y_scores)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = []
        for train_ind, val_ind in kf.split(base_dist):
            X_train, Y_train = get_data(base_dist[train_ind])
            X_test, Y_test = get_data(base_dist[val_ind])

            classifier.fit(X_train, Y_train)

            y_scores = classifier.predict(X_test)
            fpr, tpr, _ = roc_curve(Y_test, y_scores)
            roc_auc += [auc(fpr, tpr)]
        roc_auc = np.array(roc_auc)

    print(f'ROC AUC {np.mean(roc_auc)}')

    return roc_auc, fpr, tpr


def net(nfeatures, nclasses):
    return MLP(nfeatures, nclasses, layers=[64, 64])

def get_auc_net(samples, data, directory, exp_name, split=0.5):
    # TODO: make this return the AUC CV score on the data that is passed?
    sv_dir = directory + f'/{exp_name}'
    n_inliers_train = int(len(samples) * split)
    n_valid = int(0.1 * n_inliers_train / 2)
    n_test_train = int(len(data) * split)
    train_data = SupervisedDataClass(samples[:(n_inliers_train - n_valid)], data[:(n_test_train - n_valid)])
    valid_data = SupervisedDataClass(samples[(n_inliers_train - n_valid):n_inliers_train],
                                     data[(n_test_train - n_valid):n_test_train])
    test_data = SupervisedDataClass(samples[:n_inliers_train], data[:n_test_train])
    batch_size = 100
    nepochs = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier = Classifier(net, train_data.nfeatures, 1, exp_name, directory=directory,
                            activation=torch.sigmoid).to(device)

    # Make an optimizer object
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Train
    fit_classifier(classifier, train_data, valid_data, optimizer, batch_size, nepochs, device, sv_dir)

    with torch.no_grad():
        y_scores = classifier.predict(test_data.data.to(device)).cpu().numpy()
    labels_test = test_data.targets.cpu().numpy()
    fpr, tpr, _ = roc_curve(labels_test, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f'ROC AUC {roc_auc}')

    return roc_auc, fpr, tpr
