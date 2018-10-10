from q2 import makeTestPattern, computeBrief, briefLite, briefMatch, testMatch, briefRotTest, briefRotLite

import scipy.io as sio
import skimage.color
import skimage.io
import skimage.feature

# Q2.1
compareX, compareY = makeTestPattern(9,256)
sio.savemat('testPattern.mat',{'compareX':compareX,'compareY':compareY})

# Q2.2
img = skimage.io.imread('../data/chickenbroth_01.jpg')
im = skimage.color.rgb2gray(img)

# YOUR CODE: Run a keypoint detector, with nonmaximum supression
# locs holds those locations n x 2
locs = None


locs, desc = computeBrief(im,locs,compareX,compareY)

# Q2.3
locs, desc = briefLite(im)

# Q2.4
testMatch()

# Q2.5
briefRotTest()

# EC 1
briefRotTest(briefRotLite)

# EC 2 
# write it yourself!