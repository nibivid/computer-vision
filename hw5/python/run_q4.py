import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.transform
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# load the weights
# run the crops through your neural network and print them out
import pickle
import string
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
params = pickle.load(open('q3_weights.pickle','rb'))

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    # plt.imshow(bw, cmap='Greys')
    # for bbox in bboxes:
    #     minr, minc, maxr, maxc = bbox
    #     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                             fill=False, edgecolor='red', linewidth=2)
    #     plt.gca().add_patch(rect)
    # plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    test_x = []
    for bbox in bboxes:
        patch = bw[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        patch = 1 - np.transpose(patch, [1,0])  # black letter, white background
        h, w = patch.shape
        size = max(h, w)
        patch_pad = np.pad(patch, ((int((size-h)/2),int((size-h)/2)),(int((size-w)/2),int((size-w)/2))),'constant',constant_values=1.0)
        patch_pad = skimage.transform.resize(patch_pad, (28, 28))
        patch_pad = np.pad(patch_pad, ((2,2),(2,2)),'constant',constant_values=1.0)
        # plt.imshow(patch_pad, cmap='Greys')
        # plt.show()
        # pdb.set_trace()
        x = patch_pad.flatten()
        test_x.append(x)
    test_x = np.array(test_x)
    test_y = np.zeros((test_x.shape[0], 36))

    test_loss, _, test_pre = forward_network(test_x, test_y, params)
    prediction = letters[np.argmax(test_pre, axis=1)]
    # pdb.set_trace()

    plt.subplots(1, figsize=(10,6))
    plt.imshow(im1)
    for i, bbox in enumerate(bboxes):
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.text(minc, minr-10, prediction[i])
        plt.gca().add_patch(rect)
    plt.show()
