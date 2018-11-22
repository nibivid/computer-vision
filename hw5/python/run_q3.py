import numpy as np
import pdb
import scipy.io
import matplotlib.pyplot as plt
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# im = train_x[2000,:].reshape((32,32))
# plt.imshow(im, cmap='Greys')
# plt.show()
# pdb.set_trace()
# im = train_x[2000,:].reshape((32,32))
# plt.imshow(im, cmap='Greys')
# plt.show()
# pdb.set_trace()
# im = train_x[3000,:].reshape((32,32))
# plt.imshow(im, cmap='Greys')
# plt.show()
# pdb.set_trace()

max_iters = 50
# pick a batch size, learning rate
batch_size = 128
learning_rate = 2*1e-3
hidden_size = 64
# pdb.set_trace()

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')
import copy
params_orig = copy.deepcopy(params)
iter_list = []
train_loss_list = []
train_acc_list = []
valid_loss_list = []
valid_acc_list = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    # pdb.set_trace()
    for xb, yb in batches:
        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb, params,'layer1')
        probs = forward(h1, params,'output', softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs
        idx_x, idx_y = np.nonzero(yb)
        delta1[idx_x, idx_y] -= 1
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Woutput'] -= params['grad_Woutput'] * learning_rate
        params['boutput'] -= params['grad_boutput'] * learning_rate
        params['Wlayer1'] -= params['grad_Wlayer1'] * learning_rate
        params['blayer1'] -= params['grad_blayer1'] * learning_rate

    total_acc /= batch_num
    total_loss /= (batch_num * batch_size)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
        valid_loss, valid_acc, _= forward_network(valid_x, valid_y, params)
        valid_loss /= valid_x.shape[0]
        print('validation loss: {:.2f}, acc: {:.2f}'.format(valid_loss, valid_acc))
        iter_list.append(itr)
        train_loss_list.append(total_loss)
        train_acc_list.append(total_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

# run on validation set and report accuracy! should be above 75%
valid_loss, valid_acc, _= forward_network(valid_x, valid_y, params)
print('Final validation accuracy: ',valid_acc)
test_loss, test_acc, test_pre = forward_network(test_x, test_y, params)
print('Final testing accuracy: ',test_acc)

# save train + validation loss + acc
plt.figure(1)
plt.subplot(121)
plt.plot(iter_list, train_loss_list, 'b', label='Training')
plt.plot(iter_list, valid_loss_list, 'r', label='Validation')
plt.title('Training and Validation Loss')
plt.xticks(np.arange(0, iter_list[-1]+1, 5))
plt.yticks(np.arange(max(0,train_loss_list[-1]-0.1), train_loss_list[0], train_loss_list[0]/5))
plt.legend()
# pdb.set_trace()

plt.subplot(122)
plt.plot(iter_list, train_acc_list, 'b', label='Training')
plt.plot(iter_list, valid_acc_list, 'r', label='Validation')
plt.title('Training and Validation Accuracy')
plt.xticks(np.arange(0, iter_list[-1]+1, 5))
plt.yticks(np.arange(0, 1, 0.2))
plt.legend(loc='lower right')
plt.show()
pdb.set_trace()

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

F = plt.figure(1, (22, 10))
# init weight
weight_init = params_orig['Wlayer1'].reshape((32,32,-1))
vmin, vmax = weight_init.min(), weight_init.max()
grid = ImageGrid(F, 121, nrows_ncols=(8,8), axes_pad=0.1, add_all=True, label_mode="L")
for i in range(hidden_size):
    ax = grid[i]
    ax.imshow(weight_init[:,:,i], vmin=vmin, vmax=vmax, interpolation='nearest')
plt.draw()

# final weight
weight_final = params['Wlayer1'].reshape((32,32,-1))
vmin, vmax = weight_final.min(), weight_final.max()
grid = ImageGrid(F, 122, nrows_ncols=(8,8), axes_pad=0.1, add_all=True, label_mode="L")
for i in range(hidden_size):
    ax = grid[i]
    ax.imshow(weight_final[:,:,i], vmin=vmin, vmax=vmax, interpolation='nearest')
plt.draw()
plt.show()

pdb.set_trace()
# Q3.1.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
idx_gt = np.argmax(test_y, axis=1)
idx_pre = np.argmax(test_pre, axis=1)
for i in range(idx_gt.shape[0]):
    confusion_matrix[idx_gt[i], idx_pre[i]] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
