import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from LucasKanade import *

# write your script here, we recommend the above libraries for making your animation
video_path = '../data/carseq.npy'
init_rect = np.array([116.0, 59.0, 151.0, 145.0])
rect = init_rect
frame_step = 1

data = np.load(video_path)
frame_num = data.shape[2]

rect_list = [init_rect]
img0 = data[:,:,0]
for i in np.arange(1, frame_num, frame_step):
    # pdb.set_trace()
    img_cur = data[:,:,i]
    p1 = LucasKanade(img0, img_cur, init_rect, rect[0:2]-init_rect[0:2])
    rect = init_rect + np.array([p1[0], p1[1], p1[0], p1[1]])
    rect_list.append(rect)
    print(p1)

# save rects
rects = np.array(rect_list)[:,[1,0,3,2]]
pdb.set_trace()
np.save('carseqrects-wcrt.npy', rects)

# load original rects
rects_old = np.load('carseqrects.npy')

# visualization
frame = [0, 100, 200, 300, 400]
fig, axs = plt.subplots(1, 5, figsize=(10,2))
for i, k in enumerate(frame):
    img = axs[i].imshow(data[:,:,k], cmap='gray')
    rect1 = rects[k]
    rect2 = rects_old[k]
    # old red, this green
    r = patches.Rectangle((rect1[0], rect1[1]),
                           round(rect1[2] - rect1[0] + 1),
                           round(rect1[3] - rect1[1] + 1),
                           fill=False, edgecolor='g')
    axs[i].add_patch(r)
    r = patches.Rectangle((rect2[0], rect2[1]),
                           round(rect2[2] - rect2[0] + 1),
                           round(rect2[3] - rect2[1] + 1),
                           fill=False, edgecolor='r')
    axs[i].add_patch(r)
plt.show()
