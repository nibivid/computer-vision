import numpy as np
from scipy.interpolate import RectBivariateSpline
import pdb

def LucasKanade(It, It1, rect, p0=np.zeros(2)):
    # Input:
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    p1 = p0
    thres = 0.001
    height, width = It.shape
    It = RectBivariateSpline(np.arange(0, height), np.arange(0, width), It)
    It1 = RectBivariateSpline(np.arange(0, height), np.arange(0, width), It1)

    x1, y1, x2, y2 = rect
    rect_h = y2 - y1 + 1
    rect_w = x2 - x1 + 1
    Y, X = np.meshgrid(np.arange(y1,y1+rect_h), np.arange(x1,x1+rect_w))
    it = It.ev(X, Y)

    while True:
        it1 = It1.ev(X+p1[0], Y+p1[1])

        # compute error
        error = it - it1

        # compute gradient
        Gx, Gy = np.gradient(it1)
        G = np.stack([Gx.flatten(), Gy.flatten()], axis=1)

        # #Jacobian
        dwdp = np.eye(2)

        # Hessian
        A = np.dot(G, dwdp)
        H = np.dot(A.T, A)

        # dp
        dp = np.dot(np.dot(np.linalg.inv(H), A.T), error.flatten())

        # update
        p1 = p1 + dp

        # check whether smaller than thres
        if np.linalg.norm(dp) < thres:
            break

        print(np.linalg.norm(dp))
        # print(p1)

    return p1
