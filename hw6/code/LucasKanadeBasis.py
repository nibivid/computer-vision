import numpy as np
import pdb
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input:
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    p1 = np.zeros(2)
    thres = 0.01
    height, width = It.shape
    It = RectBivariateSpline(np.arange(0, height), np.arange(0, width), It)
    It1 = RectBivariateSpline(np.arange(0, height), np.arange(0, width), It1)

    x1, y1, x2, y2 = rect
    # rect_h = y2 - y1 + 1
    # rect_w = x2 - x1 + 1
    rect_h = bases.shape[1]
    rect_w = bases.shape[0]
    # pdb.set_trace()
    # print(rect_h, rect_w)
    Y, X = np.meshgrid(np.linspace(y1,y1+rect_h,rect_h), np.linspace(x1,x1+rect_w,rect_w))
    it = It.ev(X, Y)

    # pdb.set_trace()
    B = np.reshape(bases, (bases.shape[0]*bases.shape[1], bases.shape[2]))
    Gx, Gy = np.gradient(it)
    G = np.stack([Gx.flatten(), Gy.flatten()], axis=1)
    dwdp = np.eye(2)
    A = np.dot(G, dwdp)
    # pdb.set_trace()
    HA = A - np.dot(np.dot(B, B.T), A)
    H = np.dot(HA.T, HA)

    while True:
        it1 = It1.ev(X+p1[0], Y+p1[1])

        # compute error
        error = it - it1
        # pdb.set_trace()

        # compute gradient
        # Gx, Gy = np.gradient(it1)
        # G = np.stack([Gx.flatten(), Gy.flatten()], axis=1)
        #
        # # #Jacobian
        # dwdp = np.eye(2)
        #
        # # Hessian
        # A = np.dot(G, dwdp)
        # HA = A - np.dot(np.dot(B, B.T), A)
        # H = np.dot(HA.T, HA)

        # dp
        b = error.flatten()
        Hb = b - np.dot(np.dot(B, B.T), b)
        dp = np.dot(np.linalg.inv(H), np.dot(HA.T, Hb))

        # update
        p1 = p1 + dp

        # check whether smaller than thres
        if np.linalg.norm(dp) < thres:
            break

        print(np.linalg.norm(dp))
        # print(p1)

    return p1
