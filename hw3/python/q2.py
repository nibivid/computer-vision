import numpy as np
import scipy.io as sio
import skimage.feature
import matplotlib.pyplot as plt
import random
import pdb

# Q2.1
# create a 2 x nbits sampling of integers from to to patchWidth^2
# read BRIEF paper for different sampling methods
def makeTestPattern(patchWidth, nbits):
    res = None
    # YOUR CODE HERE
    # implement uniform sampling from(-S/2, S/2)
    idxMax = patchWidth * patchWidth - 1
    res = np.array([random.randint(0, idxMax+1) for i in range(2*nbits)])
    res = np.reshape(res, (2, nbits))
    # sio.savemat('testPattern.mat', {'res':res})
    return res[0,:], res[1,:]

# Q2.2
# im is 1 channel image, locs are locations
# compareX and compareY are idx in patchWidth^2
# should return the new set of locs and their descs
def computeBrief(im,locs,compareX,compareY,patchWidth=9):
    desc = None
    # YOUR CODE HERE
    h, w = im.shape[0], im.shape[1]
    radius = (patchWidth+1)//2
    desc = []
    locs_new = []
    for i in range(locs.shape[0]):
        # invalid Harris point, too close to the image boundary
        if locs[i,0]<=radius or locs[i,0]>=h-1-radius or locs[i,1]<=radius or locs[i,1]>=w-1-radius:
            continue

        # valid Harris point
        descriptor = []
        X_h, X_w = transfer_1d_2d(compareX, patchWidth)
        X_h = locs[i,0] - X_h
        X_w = locs[i,1] - X_w
        Y_h, Y_w = transfer_1d_2d(compareY, patchWidth)
        Y_h = locs[i,0] - Y_h
        Y_w = locs[i,1] - Y_w
        descriptor = 1*(im[X_h, X_w] > im[Y_h, Y_w])

        locs_new.append([locs[i,0], locs[i,1]])
        desc.append(descriptor)

    locs = np.array(locs_new)       # m x 2
    desc = np.squeeze(np.array(desc))           # m x bits
    return locs, desc

# transfer the 1D index in the patch to 2D idex, center is (0,0)
# assume the patchWidth is odd
def transfer_1d_2d(index, patchWidth=9):
    # assume the patchWidth should be odd, if not, make it to odd
    if patchWidth % 2 == 0:
        patchWidth += 1
    idx_h = index // patchWidth
    idx_w = index - idx_h * patchWidth
    idx_h -= patchWidth//2
    idx_w -= patchWidth//2
    return idx_h, idx_w

# Q2.3
# im is a 1 channel image
# locs are locations
# descs are descriptors
# if using Harris corners, use a sigma of 1.5
def briefLite(im):
    locs, desc = None, None
    # YOUR CODE HERE
    # find points
    locs = skimage.feature.corner_peaks(skimage.feature.corner_harris(im,sigma=2),min_distance=1)

    # load pattern
    data = sio.loadmat('testPattern.mat')
    compareX = data['compareX']
    compareY = data['compareY']

    # call compute brief
    locs, desc = computeBrief(im,locs,compareX,compareY,patchWidth=9)

    return locs, desc

# Q 2.4
def briefMatch(desc1,desc2,ratio=0.8):
    # okay so we say we SAY we use the ratio test
    # which SIFT does
    # but come on, I (your humble TA), don't want to.
    # ensuring bijection is almost as good
    # maybe better
    # trust me
    matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True)#,max_ratio=ratio)
    return matches

def plotMatches(im1,im2,matches,locs1,locs2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r')
    plt.show()
    return

def testMatch():
    # YOUR CODE HERE
    img1 = skimage.io.imread('../data/chickenbroth_01.jpg')
    im1 = skimage.color.rgb2gray(img1)
    locs1, desc1 = briefLite(im1)

    img2 = skimage.io.imread('../data/model_chickenbroth.jpg')
    im2 = skimage.color.rgb2gray(img2)
    locs2, desc2 = briefLite(im2)

    matches = briefMatch(desc1, desc2)
    plotMatches(im1, im2, matches, locs1, locs2)
    return


# Q 2.5
# we're also going to use this to test our
# extra-credit rotational code
def briefRotTest(briefFunc=briefLite):
    # you'll want this
    import skimage.transform
    # YOUR CODE HERE
    img = skimage.io.imread('../data/model_chickenbroth.jpg')
    im  =skimage.color.rgb2gray(img)

    # compote matches pairs
    match_num = []
    angle = []
    for i in range(0, 360, 10):
        im_rot = skimage.transform.rotate(im, i)
        locs1, desc1 = briefLite(im)
        locs2, desc2 = briefLite(im_rot)
        matches = briefMatch(desc1, desc2)
        # if i == 30:
        #     plotMatches(im, im_rot, matches, locs1, locs2)
        match_num.append(correctMatchNum(locs1, locs2, matches, i, im.shape))
        angle.append(i)
    match_num = np.array(match_num)
    print(match_num)
    angle = np.array(angle)
    print(angle)

    # draw bar plot
    plt.bar(angle, match_num)
    plt.show()
    return

# count the correct matched mun
def correctMatchNum(locs1, locs2, matches, angle, shape):
    match_num = 0
    locs1_match = locs1[matches[:,0], :]
    locs2_match = locs2[matches[:,1], :]
    locs1_match_rot = rotPosition(locs1_match, angle, shape)
    return isMatch(locs1_match_rot, locs2_match)

# find the correct corresponding location for matched points
def rotPosition(locs, angle, shape):
    ch = shape[0] // 2
    cw = shape[1] // 2
    angle = angle * np.pi / 180.
    r = np.sqrt((locs[:,0]-ch)*(locs[:,0]-ch)+(locs[:,1]-cw)*(locs[:,1]-cw))
    phi = np.arctan((ch-locs[:,0])/(cw-locs[:,1]))
    h_new = ch - r * np.sin(phi-angle)
    w_new = cw - r * np.cos(phi-angle)
    return np.transpose(np.array([h_new, w_new]))

# check whether matched
def isMatch(locs1_match_rot, locs2_match):
    dis2 = (locs1_match_rot[:,0]-locs2_match[:,0]) *(locs1_match_rot[:,0]-locs2_match[:,0]) + \
            (locs1_match_rot[:,1]-locs2_match[:,1]) *(locs1_match_rot[:,1]-locs2_match[:,1])
    match = dis2 < 100
    match_num = np.sum(match)
    return match_num

# Q2.6
# YOUR CODE HERE


# put your rotationally invariant briefLite() function here
def briefRotLite(im):
    locs, desc = None, None
    # YOUR CODE HERE

    return locs, desc
