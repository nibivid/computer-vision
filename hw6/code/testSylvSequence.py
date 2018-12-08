import numpy as np
import pdb
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import *
from LucasKanade import *

# write your script here, we recommend the above libraries for making your animation
video_path = '../data/sylvseq.npy'
base_path = '../data/sylvbases.npy'
init_rect = [61.0, 101.0, 107.0, 155.0]
frame_step = 1

data = np.load(video_path)
frame_num = data.shape[2]
bases = np.load(base_path)

rect_list = [init_rect]
for i in np.arange(1, frame_num, frame_step):
    # pdb.set_trace()
    img_pre = data[:,:,i-1]
    img_cur = data[:,:,i]
    p1 = LucasKanadeBasis(img_pre, img_cur, init_rect, bases)
    init_rect = init_rect + np.array([p1[0], p1[1], p1[0], p1[1]])
    rect_list.append(init_rect)
    print(p1)

init_rect = [61.0, 101.0, 107.0, 155.0]
rect_list_old = [init_rect]
for i in np.arange(1, frame_num, frame_step):
    # pdb.set_trace()
    img_pre = data[:,:,i-1]
    img_cur = data[:,:,i]
    p1 = LucasKanade(img_pre, img_cur, init_rect)
    init_rect = init_rect + np.array([p1[0], p1[1], p1[0], p1[1]])
    rect_list_old.append(init_rect)
    print(p1)

# save rects
rects = np.array(rect_list)[:,[1,0,3,2]]
np.save('sylvseqrects.npy', rects)
rects_old = np.array(rect_list_old)[:,[1,0,3,2]]
np.save('sylvseqrects-old.npy', rects_old)

# visualization
frame = [0, 100, 200, 300, 400]
fig, axs = plt.subplots(1, 5, figsize=(10,2))
for i, k in enumerate(frame):
    img = axs[i].imshow(data[:,:,k], cmap='gray')
    rect = rect_list_old[k]
    r = patches.Rectangle((rect[1], rect[0]),
                           round(rect[3] - rect[1] + 1),
                           round(rect[2] - rect[0] + 1),
                           fill=False, edgecolor='r')
    axs[i].add_patch(r)
    rect = rect_list[k]
    r = patches.Rectangle((rect[1], rect[0]),
                           round(rect[3] - rect[1] + 1),
                           round(rect[2] - rect[0] + 1),
                           fill=False, edgecolor='g')
    axs[i].add_patch(r)
plt.show()
