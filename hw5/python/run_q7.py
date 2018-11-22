import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import scipy
import pdb
import scipy.io
from nn_pytorch import *
import matplotlib.pyplot as plt


# string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]

# q7.1
if False:
    batch_size = 128
    epochs = 50
    hidden_size = 64
    lr = 0.02
    train_nist36 = Dataset_Loader(dataset_name='nist36', set_name='train', batch_size=batch_size)
    fully_connected_net = Fully_Connected_Network(hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(fully_connected_net.parameters(), lr=lr)

    iter_list = []
    train_loss_list = []
    train_acc_list = []
    batch_num = len(train_nist36)//batch_size
    # pdb.set_trace()
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for idx in range(batch_num):
            # pdb.set_trace()
            X, Y = train_nist36[idx]['image'], train_nist36[idx]['label']
            optimizer.zero_grad()
            probs = fully_connected_net(Variable(torch.Tensor(X), requires_grad=True))
            loss = criterion(probs, Variable(torch.Tensor(Y)).long())
            loss.backward()
            optimizer.step()
            # pdb.set_trace()
            total_loss += loss.data[0]
            total_acc += compute_accuracy(Y, probs)

        total_acc /= batch_num
        total_loss /= batch_num

        if epoch % 2 == 0:
            print("epoch: {:02d} \t loss: {:.4f} \t acc : {:.2f}".format(epoch,total_loss,total_acc))
            iter_list.append(epoch)
            train_loss_list.append(total_loss)
            train_acc_list.append(total_acc)

# q7.2
if False:
    batch_size = 32
    epochs = 3
    lr = 0.001
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_mnist = torchvision.datasets.MNIST('../data/mnist', train=True, download=True, transform=transform)
    train_mnist_loader = torch.utils.data.DataLoader(train_mnist, batch_size=batch_size,
                                                    shuffle=True, num_workers=2)
    convolution_net = Convolution_Network(num_cls=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(convolution_net.parameters(), lr=lr)

    iter_list = []
    train_loss_list = []
    train_acc_list = []
    batch_num = len(train_mnist_loader)
    # pdb.set_trace()
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        # pdb.set_trace()
        for idx, data in enumerate(train_mnist_loader):
            X, Y = data
            # pdb.set_trace()
            optimizer.zero_grad()
            probs = convolution_net(Variable(torch.Tensor(X), requires_grad=True))
            # pdb.set_trace()
            loss = criterion(probs, Variable(Y))
            loss.backward()
            optimizer.step()
            # pdb.set_trace()
            total_loss += loss.data[0]
            total_acc += compute_accuracy_tensor(Y, probs)

            if idx % 200 == 199:
                print("iter: {:02d} \t loss: {:.4f} \t acc : {:.2f}".format(idx,total_loss/(idx+1),total_acc/(idx+1)))
                iter_list.append(epoch*len(train_mnist_loader)+idx)
                train_loss_list.append(total_loss/(idx+1))
                train_acc_list.append(total_acc/(idx+1))

# q7.3
if False:
    batch_size = 32
    epochs = 8
    lr = 0.01
    train_nist36 = Dataset_Loader(dataset_name='nist36', set_name='train', batch_size=batch_size, reshape=True)
    convolution_net = Convolution_Network(num_cls=36)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(convolution_net.parameters(), lr=lr)

    iter_list = []
    train_loss_list = []
    train_acc_list = []
    batch_num = len(train_nist36)//batch_size
    # pdb.set_trace()
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for idx in range(batch_num):
            # pdb.set_trace()
            X, Y = train_nist36[idx]['image'], train_nist36[idx]['label']
            optimizer.zero_grad()
            probs = convolution_net(Variable(torch.Tensor(X), requires_grad=True))
            # pdb.set_trace()
            loss = criterion(probs, Variable(torch.Tensor(Y)).long())
            loss.backward()
            optimizer.step()
            # pdb.set_trace()
            total_loss += loss.data[0]
            total_acc += compute_accuracy(Y, probs)

            if idx % 100 == 99:
                print("iter: {:02d} \t loss: {:.4f} \t acc : {:.2f}".format(idx,total_loss/(idx+1),total_acc/(idx+1)))
                iter_list.append(epoch*batch_num+idx)
                train_loss_list.append(total_loss/(idx+1))
                train_acc_list.append(total_acc/(idx+1))

# q7.4
# if True:
#     batch_size = 32
#     epochs = 3
#     lr = 0.001
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#     train_emnist = torchvision.datasets.EMNIST('../data/emnist', train=True, download=True, transform=transform)
#     train_emnist_loader = torch.utils.data.DataLoader(train_emnist, batch_size=batch_size,
#                                                     shuffle=True, num_workers=2)
#     pdb.set_trace()
#     convolution_net = Convolution_Network(num_cls=36)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(convolution_net.parameters(), lr=lr)
#
#     iter_list = []
#     train_loss_list = []
#     train_acc_list = []
#     batch_num = len(train_emnist_loader)
#     # pdb.set_trace()
#     for epoch in range(epochs):
#         total_loss = 0
#         total_acc = 0
#         # pdb.set_trace()
#         for idx, data in enumerate(train_mnist_loader):
#             X, Y = data
#             # pdb.set_trace()
#             optimizer.zero_grad()
#             probs = convolution_net(Variable(torch.Tensor(X), requires_grad=True))
#             # pdb.set_trace()
#             loss = criterion(probs, Variable(Y))
#             loss.backward()
#             optimizer.step()
#             # pdb.set_trace()
#             total_loss += loss.data[0]
#             total_acc += compute_accuracy_tensor(Y, probs)
#
#             if idx % 200 == 199:
#                 print("iter: {:02d} \t loss: {:.4f} \t acc : {:.2f}".format(idx,total_loss/(idx+1),total_acc/(idx+1)))
#                 iter_list.append(epoch*len(train_mnist_loader)+idx)
#                 train_loss_list.append(total_loss/(idx+1))
#                 train_acc_list.append(total_acc/(idx+1))

# save train + validation loss + acc
plt.figure(1)
plt.subplot(121)
plt.plot(iter_list, train_loss_list, 'b')
plt.title('Training Loss')
plt.xticks(np.arange(0, iter_list[-1]+1, 1000))
plt.yticks(np.arange(max(0,train_loss_list[-1]-0.1), train_loss_list[0], train_loss_list[0]/5))
plt.legend()
# pdb.set_trace()

plt.subplot(122)
plt.plot(iter_list, train_acc_list, 'b')
plt.title('Training Accuracy')
plt.xticks(np.arange(0, iter_list[-1]+1, 1000))
plt.yticks(np.arange(0, 1, 0.2))
plt.legend(loc='lower right')
plt.show()
pdb.set_trace()
