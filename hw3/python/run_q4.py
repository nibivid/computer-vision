from q4 import *
from q3 import *
from q2 import *
import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import skimage.io

# Q 4.1
# load images into img1
# and img2
# compute H2to1
# please use any feature method you like that gives
# good results
# img1 = skimage.io.imread('../data/pnc1.png')
# img2 = skimage.io.imread('../data/pnc0.png')
# img1 = skimage.io.imread('../data/incline_L.png')
# img2 = skimage.io.imread('../data/incline_R.png')
# img1 = skimage.io.imread('../data/self_L.png')
# img2 = skimage.io.imread('../data/self_R.png')
img1 = skimage.io.imread('../data/self_2.png')
img2 = skimage.io.imread('../data/self_3.png')

bestH2to1 = None
# YOUR CODE HERE
# bestH2to1 = findH(img1, img2)

# panoImage = imageStitching(img1,img2,bestH2to1)
# plt.subplot(1,2,1)
# plt.imshow(img1)
# plt.title('pnc0')
# plt.subplot(1,2,2)
# plt.title('pnc1')
# plt.imshow(img2)
# plt.figure()
# plt.imshow(panoImage)
# plt.show()

# Q 4.2
# panoImage2= imageStitching_noClip(img1,img2,bestH2to1)
# plt.subplot(1,2,1)
# plt.imshow(panoImage)
# plt.subplot(1,2,2)
# plt.imshow(panoImage2)
# plt.show()

# Q 4.3
# panoImage3 = generatePanorama(img1, img2)
# plt.imshow(panoImage3)
# plt.show()

# Q 4.4 (EC)
# Stitch your own photos with your code

# Q 4.5 (EC)
# Write code to stitch multiple photos
# see http://www.cs.jhu.edu/~misha/Code/DMG/PNC3/PNC3.zip
# for the full PNC dataset if you want to use that
pdb.set_trace()
imgs = [skimage.io.imread('../data/self_{}.png'.format(i)) for i in range(1,7)]
panoImage4 = generateMultiPanorama(imgs)
plt.imshow(panoImage4)
plt.show()

if False:
    imgs = [skimage.io.imread('../PNC3/src_000{}.png'.format(i)) for i in range(7)]
    panoImage4 = generateMultiPanorama(imgs)
    plt.imshow(panoImage4)
    plt.show()
