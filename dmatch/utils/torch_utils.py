import torch
import numpy as np


def get_numpy(x):
    if torch.is_tensor(x):
        x = x.detach()
        if torch.cuda.is_available():
            x = x.cpu()
        x = x.numpy()
    return x

def batch_function(method, data, batch_size):
    n_data = len(data)
    n_batches = int(np.ceil(len(data) / batch_size))
    n_over = int(n_batches * batch_size - n_data )
    data = torch.cat((data, torch.zeros(n_over, data.shape[1]).to(data.device)))
    data = data.reshape(n_batches, batch_size, data.shape[1])
    outputs = []
    for i in range(n_batches):
        outputs += [method(data[i])]
    return torch.cat(outputs)[:n_data]


def batch_predict(model, data_array, encode=False):
    store = []
    for data in data_array:
        if encode:
            store += [torch.cat(model.encode(data), 1)]
        else:
            store += [model(data)]
    return torch.cat(store)