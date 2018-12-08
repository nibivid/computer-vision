import numpy as np
from scipy.ndimage import affine_transform
# from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_opening, disk
import skimage.morphology
from LucasKanadeAffine import *

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1
	# Output:
	#	mask: [nxm]
    # put your implementation here
    thres = 0.08
    M = LucasKanadeAffine(image1, image2)
    # warp
    # pdb.set_trace()
    image1_warp = affine_transform(image1, M[:2,:2])
    diff = np.abs(image1_warp - image2)
    mask = diff > thres
    # pdb.set_trace()
    # struct = skimage.morphology.disk(5)
    # mask = skimage.morphology.binary_opening(mask).astype(bool)
    mask[:10,:] = 0
    mask[-10:,:] = 0
    mask[:, :10] = 0
    mask[:, -10:] = 0
    # struct = skimage.morphology.disk(2)
    # mask = skimage.morphology.binary_erosion(mask, selem=struct)
    struct = skimage.morphology.disk(2)
    mask = skimage.morphology.binary_dilation(mask, selem=struct)
    mask = skimage.morphology.remove_small_objects(mask, min_size=10)
    mask = skimage.morphology.remove_small_holes(mask, min_size=200)
    struct = skimage.morphology.disk(3)
    mask = skimage.morphology.binary_erosion(mask, selem=struct)
    # mask = skimage.morphology.remove_small_objects(mask, min_size=10)

    return mask
