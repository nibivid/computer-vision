import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate =  5e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights_with_momentum(1024,hidden_size,params,'layer1')
initialize_weights_with_momentum(hidden_size,hidden_size,params,'hidden')
initialize_weights_with_momentum(hidden_size,hidden_size,params,'hidden2')
initialize_weights_with_momentum(hidden_size,1024,params,'output')

iter_list = []
train_loss_list = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        h1 = forward(xb, params,'layer1', relu)
        h2 = forward(h1, params,'hidden', relu)
        h3 = forward(h2, params,'hidden2', relu)
        yb = forward(h3, params,'output', sigmoid)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss = np.sum((xb - yb)**2)
        total_loss += loss

        # backward
        delta1 = -2 * (xb - yb)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'hidden2', relu_deriv)
        delta4 = backwards(delta3, params, 'hidden', relu_deriv)
        backwards(delta4, params, 'layer1', relu_deriv)

        # momentum, update gradient
        # pdb.set_trace()
        for k, v in params.items():
            if '_' in k:
                continue
            # params[k] -= params['grad_' + k] * learning_rate
            params['mom_' + k] = 0.9 * params['mom_' + k] - params['grad_' + k] * learning_rate
            params[k] += params['mom_' + k]

    total_loss /= (batch_num * batch_size)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.4f}".format(itr,total_loss))
        iter_list.append(itr)
        train_loss_list.append(total_loss)

    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# save model
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q5_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 5.2
# save train loss
plt.figure(1)
plt.plot(iter_list, train_loss_list, 'b')
plt.title('Training Loss')
plt.xticks(np.arange(0, iter_list[-1]+1, 5))
plt.yticks(np.arange(max(0,train_loss_list[-1]-1), train_loss_list[0], (train_loss_list[0]-train_loss_list[-1])/5))
plt.show()

# visualize some results
# Q5.3.1
import matplotlib.pyplot as plt
pdb.set_trace()
# X = [[valid_x[100*i,:],valid_x[100*i+1,:]] for i in range(5)]
# X = np.array(X)
# X = np.reshape(X, [10,-1])

h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
# pdb.set_trace()
for i in range(5):
    plt.subplot(2,2,1)
    plt.imshow(valid_x[100*i,:].reshape(32,32).T)
    plt.subplot(2,2,2)
    plt.imshow(out[100*i,:].reshape(32,32).T)
    plt.subplot(2,2,3)
    plt.imshow(valid_x[100*i+1,:].reshape(32,32).T)
    plt.subplot(2,2,4)
    plt.imshow(out[100*i+1,:].reshape(32,32).T)
    plt.show()
pdb.set_trace()

from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2
MSE = np.sum((valid_x - out)**2, axis=1) / 1024
MAX = np.max(out, axis=1)
PSNR = 20 * np.log10(MAX) - 10 * np.log10(MSE)
psnr_mean = np.mean(PSNR)
print('Average PSNR: {}'.format(psnr_mean))
pdb.set_trace()
psnr_skimage = psnr(valid_x, out, data_range=None)
print('Average PSNR from Skimage: {}'.format(psnr_skimage))
