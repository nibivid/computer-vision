import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib import animation
import matplotlib.patches as patches

from SubtractDominantMotion import *

# write your script here, we recommend the above libraries for making your animation
video_path = '../data/aerialseq.npy'
frame_step = 1

data = np.load(video_path)
frame_num = data.shape[2]

masks = np.zeros(data.shape)
for i in np.arange(1, frame_num, frame_step):
    # pdb.set_trace()
    img_pre = data[:,:,i-1]
    img_cur = data[:,:,i]
    mask = SubtractDominantMotion(img_pre, img_cur)
    masks[:,:,i] = mask
    # plt.imshow(data[:,:,i], cmap='gray')
    # plt.imshow(mask, cmap='Reds', alpha=0.4)
    # plt.show()
    # pdb.set_trace()

# visualization
frame = [30, 60, 90, 120]
fig, axs = plt.subplots(1, 4, figsize=(8,2))
for i, k in enumerate(frame):
    axs[i].imshow(data[:,:,k], cmap='gray')
    axs[i].imshow(masks[:,:,k], cmap='Reds', alpha=0.4)
plt.show()
