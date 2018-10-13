from q2 import makeTestPattern, computeBrief, briefLite, briefMatch, testMatch, briefRotTest, briefRotLite

import scipy.io as sio
import skimage.color
import skimage.io
import skimage.feature
import pdb

# Q2.1
# pdb.set_trace()
compareX, compareY = makeTestPattern(9,256)
sio.savemat('testPattern.mat',{'compareX':compareX,'compareY':compareY})

# Q2.2
# pdb.set_trace()
img = skimage.io.imread('../data/chickenbroth_01.jpg')
im = skimage.color.rgb2gray(img)

# YOUR CODE: Run a keypoint detector, with nonmaximum supression
# locs holds those locations n x 2
locs = None

# extract Harris points
# locs = skimage.feature.corner_peaks(skimage.feature.corner_harris(im,sigma=1),min_distance=1)
# print(locs)

# draw Harris points
# for i in range(locs.shape[0]):
#     rr,cc = skimage.draw.circle_perimeter(locs[i,0], locs[i,1], 2)
#     im[rr,cc] = 1
# skimage.io.imshow(im)
# skimage.io.show()

# pdb.set_trace()
# locs, desc = computeBrief(im,locs,compareX,compareY)

# Q2.3
# locs, desc = briefLite(im)

# Q2.4
# testMatch()

# Q2.5
pdb.set_trace()
briefRotTest()

# EC 1
pdb.set_trace()
briefRotTest(briefRotLite)

# EC 2
# write it yourself!
