from torch.utils.data import Dataset
import torch

class SupervisedDataClass(Dataset):
    def __init__(self, inliers, outliers):
        super(SupervisedDataClass, self).__init__()
        self.data = torch.cat((inliers, outliers), 0)
        self.targets = torch.cat((torch.ones(len(inliers)), torch.zeros(len(outliers))), 0).view(-1, 1)
        self.nfeatures = self.data.shape[1]

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return self.data.shape[0]