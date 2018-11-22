import pdb
import numpy as np
import torch
import scipy
import torch.nn as nn
import torch.nn.functional as F

class Dataset_Loader(torch.utils.data.Dataset):
    def __init__(self, dataset_name, set_name, batch_size, reshape=False):
        self.dataset_name = dataset_name
        self.set_name = set_name
        self.batch_size = batch_size
        self.reshape = reshape
        self.data = scipy.io.loadmat('../data/' + self.dataset_name + '_' + self.set_name + '.mat')
        self.X, self.Y = self.data[self.set_name + '_data'], self.data[self.set_name + '_labels']
        self.shuffle()
        self.Y = np.argmax(self.Y, axis=1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {'image':self.X[idx:idx+self.batch_size,:], 'label':self.Y[idx:idx+self.batch_size]}

    def shuffle(self):
        num = self.X.shape[0]
        idx_order = np.random.permutation(num)
        self.X = self.X[idx_order]
        self.Y = self.Y[idx_order]
        if self.reshape:
            self.X = np.reshape(self.X, [-1,1,32,32])
            self.X = self.X[:,:,2:-2, 2:-2]

class Fully_Connected_Network(nn.Module):
    def __init__(self, hidden_size):
        super(Fully_Connected_Network, self).__init__()
        self.layer1 = nn.Linear(1024, hidden_size)
        self.output = nn.Linear(hidden_size, 36)

    def forward(self, input):
        # pdb.set_trace()
        x = F.sigmoid(self.layer1(input))
        x = self.output(x)
        return x

class Convolution_Network(nn.Module):
    def __init__(self, num_cls):
        super(Convolution_Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_cls)


    def forward(self, input):
        # pdb.set_trace()
        x = F.relu(F.max_pool2d(self.conv1(input), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # pdb.set_trace()
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def compute_accuracy(gt, pred):
    pred = np.argmax(pred.data.numpy(), axis=1)
    # pdb.set_trace()
    sum = np.sum((gt == pred))
    return sum*1.0 / gt.shape[0]

def compute_accuracy_tensor(gt, pred):
    pred = np.argmax(pred.data.numpy(), axis=1)
    # pdb.set_trace()
    sum = np.sum((gt.numpy() == pred))
    return sum*1.0 / gt.shape[0]
