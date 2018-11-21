import numpy as np
import pdb
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
U, S, Vh = np.linalg.svd(np.dot(train_x.T, train_x))
pdb.set_trace()
#projection matrix
P = U[:, :dim]

# rebuild a low-rank version
lrank = None
lrank = np.dot(valid_x, P)

# rebuild it
recon = None
recon = np.dot(lrank, P.T)

for i in range(5):
    plt.subplot(2,2,1)
    plt.imshow(valid_x[100*i,:].reshape(32,32).T)
    plt.subplot(2,2,2)
    plt.imshow(recon[100*i,:].reshape(32,32).T)
    plt.subplot(2,2,3)
    plt.imshow(valid_x[100*i+1,:].reshape(32,32).T)
    plt.subplot(2,2,4)
    plt.imshow(recon[100*i+1,:].reshape(32,32).T)
    plt.show()

# build valid dataset
recon_valid = None
recon_valid = recon

total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())
