import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import pdb

from LucasKanade import *

# write your script here, we recommend the above libraries for making your animation
video_path = '../data/carseq.npy'
init_rect = [116.0, 59.0, 151.0, 145.0]
frame_step = 1

data = np.load(video_path)
frame_num = data.shape[2]

rect_list = [init_rect]
for i in np.arange(1, frame_num, frame_step):
    # pdb.set_trace()
    img_pre = data[:,:,i-1]
    img_cur = data[:,:,i]
    p1 = LucasKanade(img_pre, img_cur, init_rect)
    init_rect = init_rect + np.array([p1[0], p1[1], p1[0], p1[1]])
    rect_list.append(init_rect)
    print(p1)

# save rects
rects = np.array(rect_list)[:,[1,0,3,2]]
pdb.set_trace()
np.save('carseqrects.npy', rects)

# visualization
frame = [0, 100, 200, 300, 400]
fig, axs = plt.subplots(1, 5, figsize=(10,2))
for i, k in enumerate(frame):
    img = axs[i].imshow(data[:,:,k], cmap='gray')
    rect = rect_list[k]
    r = patches.Rectangle((rect[1], rect[0]),
                           round(rect[3] - rect[1] + 1),
                           round(rect[2] - rect[0] + 1),
                           fill=False, edgecolor='r')
    axs[i].add_patch(r)
plt.show()
