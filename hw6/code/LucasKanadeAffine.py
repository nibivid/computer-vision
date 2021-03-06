import numpy as np
import pdb
from scipy.interpolate import RectBivariateSpline

def valid_pts(pts, height, width):
    # idx = []
    idx = (pts[0,:]>0) & (pts[0,:]<width) & (pts[1,:]>0) & (pts[1,:]<height)
    idx = np.where(idx == True)
    return idx[0]

def LucasKanadeAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    # M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M = np.eye(3)
    p1 = np.zeros(6)
    thres = 0.1
    # define pts
    height, width = It.shape
    Y, X = np.meshgrid(np.arange(0, height), np.arange(0, width))
    pts = np.stack([X.flatten().T, Y.flatten().T, np.ones((X.shape[0]*X.shape[1]))], axis=0)

    # define spline
    It = RectBivariateSpline(np.arange(0, height), np.arange(0, width), It)
    It1 = RectBivariateSpline(np.arange(0, height), np.arange(0, width), It1)
    it = It.ev(X, Y)

    # compute gradient
    Gx, Gy = np.gradient(it)
    Gx = Gx.flatten()
    Gy = Gy.flatten()
    it = it.flatten()
    # G = np.stack([Gx.flatten(), Gy.flatten()], axis=1)

    # start loop
    while True:
        M = np.array([[1+p1[0], p1[2], p1[4]], [p1[1], 1+p1[3], p1[5]], [0,0,1]]).astype(float)
        pts1 = np.dot(M, pts)
        it1 = It1.ev(pts1[0,:], pts1[1,:])
        # check whether point inside the image
        valid_idx = valid_pts(pts1, height, width)

        # error
        error = it - it1
        error = error[valid_idx]

        # SD
        SD = np.array([Gx*pts1[0,:].T, Gy*pts1[0,:].T, Gx*pts1[1,:], Gy*pts1[1,:], Gx, Gy])
        SDvalid = SD[:, valid_idx]

        # build Hessian
        H = np.dot(SDvalid, SDvalid.T)

        # delta p
        dp = np.dot(np.dot(np.linalg.inv(H), SDvalid), error.flatten())

        # update
        p1 = p1 + dp

        # pdb.set_trace()
        # check whether smaller than thres
        print(np.linalg.norm(dp))
        if np.linalg.norm(dp) < thres:
            M = np.array([[1+p1[0], p1[2], p1[4]], [p1[1], 1+p1[3], p1[5]], [0,0,1]]).astype(float)
            break


    return M
