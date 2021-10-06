import torch.nn as nn
import torch
import numpy as np

# from jets_utils.jutils.models.pytorch_models import BasicPytorchModel

class flow_builder(nn.Module):
    def __init__(self, flow, base_dist, device, name, directory='INN_test'):
        super(flow_builder, self).__init__()
        self.directory = directory
        self.exp_name = name
        self.flow = flow
        self.base_dist = base_dist
        self.device = device
        self.loss_names = ['mle']
        self.dir = directory

    # Semantically within the scope of this project this should return the latent space.
    def forward(self, data):
        return self.encode(data)

    def encode(self, x):
        return self.flow.transform_to_noise(x)

    def decode(self, x):
        return self.flow._transform.inverse(x)[0]

    def autoencode(self, data):
        print('Encoding and decoding is irrelevant for invertible models')
        return data

    def save(self, path):
        torch.save(self.flow.state_dict(), path)

    def load(self, path, device=None):
        if device:
            self.flow.load_state_dict(torch.load(path, map_location=device))
        else:
            self.flow.load_state_dict(torch.load(path))

    def sample(self, num, batch_size=None):
        return self.flow.sample(num)

    def get_numpy(self, x):
        if torch.is_tensor(x):
            x = x.detach()
            if self.device != 'cpu':
                x = x.cpu()
            x = x.numpy()
        return x

    def get_det_J(self, data):
        return self.flow._transform(data)[1]

    def log_prob(self, data, batch_size=None):
        if batch_size:
            n_full = int(data.shape[0] // batch_size)
            n_features = data.shape[1]
            bd = data[:n_full * batch_size].view(-1, batch_size, n_features)
            n_batches = bd.shape[0]
            op = torch.empty((n_batches, batch_size))
            for i in range(n_batches):
                op[i] = self.flow.log_prob(bd[i])
            op = op.view(-1, 1)
            op = torch.cat((op, self.flow.log_prob(data[n_full * batch_size:]).view(-1, 1)), 0)
            return op.view(-1)
        else:
            return self.flow.log_prob(data)

    def compute_loss(self, data, batch_size):
        self.mle = -self.flow.log_prob(data).mean()
        return self.mle

    def get_loss_state(self, nsf=10):
        return {'mle': self.mle.item()}

    def get_scores(self, data):
        # Save the calculations to avoid redundant computations
        self.scores = {'MLE': self.log_prob(data)}
        return self.scores

    def get_outputs(self, data, batch_size, *args, **kwargs):
        n_outputs = len(self.forward(data[:1]))
        outputs = [[]] * n_outputs
        n_data = len(data)
        n_batches = int(np.ceil(len(data) / batch_size))
        n_over = int(n_batches * batch_size - n_data )
        data = torch.cat((data, torch.zeros(n_over, data.shape[1]).to(data.device)))
        data = data.reshape(n_batches, batch_size, data.shape[1])
        for i in range(n_batches):
            outs = self.forward(data[i], **kwargs)
            if n_outputs > 1:
                for j in range(n_outputs):
                    outputs[j] += outs[j]
            else:
                outputs[0] += [outs]
        finals = []
        for out in outputs:
            finals += [torch.cat(out)[:n_data]]
        return finals
