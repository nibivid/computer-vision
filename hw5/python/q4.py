import numpy as np
import pdb

import skimage
import skimage.io
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions
    image_blur = skimage.filters.gaussian(image, sigma=1)
    image_gray = skimage.color.rgb2gray(image_blur)
    thresh = np.mean(image_gray) * 5 / 7.0
    image_thresh = (image_gray < thresh).astype(float)
    # image_morph = skimage.morphology.opening(image_thresh, skimage.morphology.square(2))
    image_morph = skimage.morphology.dilation(image_thresh, skimage.morphology.square(4))
    image_label = skimage.measure.label(image_morph, connectivity=2)

    # get bboxes
    bboxes_tmp = bboxes
    # fig, ax = plt.subplots(1, figsize=(10,6))
    # ax.imshow(image_morph, cmap='Greys')
    area_mean = 0
    for region in skimage.measure.regionprops(image_label):
        # take regions with large enough areas
        if region.area >= 200:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            area_mean += (maxr - minr) * (maxc - minc)
            bboxes_tmp.append([minr, minc, maxr, maxc])
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
    #         ax.add_patch(rect)
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    area_mean /= (1.0 * len(bboxes_tmp))

    # combine bboxes together
    for i, box_i in enumerate(bboxes_tmp[:-2]):
        for j, box_j in enumerate(bboxes_tmp[i+1:]):
            if sameLetter(box_i, box_j, area_mean):
                minr = min(box_i[0], box_j[0])
                minc = min(box_i[1], box_j[1])
                maxr = max(box_i[2], box_j[2])
                maxc = max(box_i[3], box_j[3])
                try:
                    bboxes.remove(box_i)
                except ValueError:
                    pass
                try:
                    bboxes.remove(box_j)
                except ValueError:
                    pass
                bboxes.append([minr, minc, maxr, maxc])

    # plot bbox
    # fig, ax = plt.subplots(1, figsize=(10,6))
    # ax.imshow(image_morph, cmap='Greys')
    # for box in bboxes:
    #     minr, minc, maxr, maxc = box[0], box[1], box[2], box[3]
    #     rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                               fill=False, edgecolor='red', linewidth=2)
    #     ax.add_patch(rect)
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    # pdb.set_trace()

    return bboxes, image_morph

# whether belong to same letter
def sameLetter(box_i, box_j, area_mean):
    # if close & area difference too much, assume as the same character
    area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
    area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
    if (area_i < 0.5 * area_j or area_j < 0.5 * area_i) and (area_i < 0.3 * area_mean or area_j < 0.3 * area_mean):
        inter_minr = max(box_i[0], box_j[0])
        inter_maxr = min(box_i[2], box_j[2])
        inter_minc = max(box_i[1], box_j[1])
        inter_maxc = min(box_i[3], box_j[3])
        thres = 10
        # inside or intersection or close
        if inter_minr <= inter_maxr + thres and inter_minc <= inter_maxc + thres:
            # pdb.set_trace()
            return True
    return False
