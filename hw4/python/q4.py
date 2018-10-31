import numpy as np
import scipy.ndimage.filters
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import skimage.io

from q2 import eightpoint
from q3 import essentialMatrix, triangulate
from util import camera2

# Q 4.1
# np.pad may be helpfuls
def epipolarCorrespondence(im1, im2, F, x1, y1):
    x2, y2 = 0, 0
    # define params.
    patchWidth = 7
    halfWidth = (patchWidth - 1) // 2
    searchRange = 50

    # define homogenous point 1 and epipolar lines
    X1 = np.array([x1, y1, 1])
    epipolarLines = np.dot(F, X1)
    # pdb.set_trace()
    # patch = im1[y1-halfWidth:y1+halfWidth+1,x1-halfWidth:x1+halfWidth+1, :]
    patch = im1[int(max(0,y1-halfWidth)):int(min(y1+halfWidth+1,im1.shape[0]-1)),
                int(max(0,x1-halfWidth)):int(min(x1+halfWidth+1,im1.shape[1]-1)), :]

    # define start+end, start loop
    start_x = max(halfWidth, x1-searchRange)
    end_x = min(im2.shape[1]-halfWidth-1, x1+searchRange)
    minErr = 10000
    for test_x in range(start_x, end_x):
        test_y = int((-epipolarLines[0]*test_x - epipolarLines[2]) / epipolarLines[1])
        if test_y >= halfWidth and test_y <= im2.shape[0]-halfWidth-1:
            testPatch = im2[test_y-halfWidth:test_y+halfWidth+1, test_x-halfWidth:test_x+halfWidth+1, :]
            # patch_diff = scipy.ndimage.filters.gaussian_filter(patch - testPatch, 1)
            patch_diff = patch - testPatch
            err1 = np.linalg.norm(patch_diff) / (patchWidth*patchWidth)
            err2 = np.linalg.norm(np.array([test_x,test_y]) - np.array([x1,y1]))
            if err1 + err2 < minErr:
                # pdb.set_trace()
                minErr = err1 + err2
                x2 = test_x
                y2 = test_y

    return x2, y2

# Q 4.2
# this is the "all in one" function that combines everything
def visualize(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2):
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.set_xlim/set_ylim/set_zlim/
    # ax.set_aspect('equal')
    # may be useful
    # you'll want a roughly cubic meter
    # around the center of mass

    # load image and pts pairs
    im1 = skimage.io.imread(IM1_PATH)
    im2 = skimage.io.imread(IM2_PATH)
    corrs = scipy.io.loadmat(TEMPLE_CORRS)
    x1 = corrs['x1']
    y1 = corrs['y1']
    pts1 = np.concatenate([x1, y1], axis=1) # (num, 2)

    # compute F and x2,y2
    pts2 = []
    for idx in range(len(x1)):
        x2, y2 = epipolarCorrespondence(im1,im2,F,int(x1[idx]),int(y1[idx]))
        pts2.append([x2, y2])
    pts2 = np.array(pts2)
    # pdb.set_trace()

    # compute E and triangulate
    E = essentialMatrix(F,K1,K2)
    E = E/E[2,2]
    pdb.set_trace()
    M2s = camera2(E)
    C1 = np.hstack([np.eye(3),np.zeros((3,1))])
    # print(M2s)
    for C2 in M2s:
        P, err = triangulate(K1.dot(C1),pts1,K2.dot(C2),pts2)
        if(P.min(0)[2] > 0):
            break
    scipy.io.savemat('q4_2.mat', {'F':F, 'M1':C1, 'M2':C2, 'C1':np.dot(K1,C1), 'C2':np.dot(K2,C2)})
    ax.scatter(P[:,0], P[:,1], P[:,2])
    plt.show()
    print('M2')
    print(C2)
    print('C2')
    print(np.dot(K2,C2))


# Extra credit
def visualizeDense(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2):
    fig = plt.figure()
    ax = Axes3D(fig)

    # load image and pts pairs
    im1 = skimage.io.imread(IM1_PATH)
    im2 = skimage.io.imread(IM2_PATH)
    corrs = scipy.io.loadmat(TEMPLE_CORRS)

    # make pts1
    x1 = []
    y1 = []
    for x in range(im1.shape[1]):
        for y in range(im2.shape[0]):
            if np.mean(im1[y, x, :] > 100):
                x1.append(x)
                y1.append(y)
    pts1 = np.stack([x1, y1], axis=1) # (num, 2)

    # compute F and x2,y2
    pts2 = []
    for idx in range(len(x1)):
        x2, y2 = epipolarCorrespondence(im1,im2,F,int(x1[idx]),int(y1[idx]))
        pts2.append([x2, y2])
    pts2 = np.array(pts2)
    # pdb.set_trace()

    # compute E and triangulate
    E = essentialMatrix(F,K1,K2)
    E = E/E[2,2]
    M2s = camera2(E)
    C1 = np.hstack([np.eye(3),np.zeros((3,1))])
    # print(M2s)
    for C2 in M2s:
        P, err = triangulate(K1.dot(C1),pts1,K2.dot(C2),pts2)
        if(P.min(0)[2] > 0):
            break
    scipy.io.savemat('q4_3.mat', {'F':F, 'M1':C1, 'M2':C2, 'C1':np.dot(K1,C1), 'C2':np.dot(K2,C2)})
    ax.scatter(P[:,0], P[:,1], P[:,2])
    plt.show()
    print('M2')
    print(C2)
    print('C2')
    print(np.dot(K2,C2))
    return
