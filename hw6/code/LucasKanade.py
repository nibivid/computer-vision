import numpy as np
from scipy.interpolate import RectBivariateSpline
import pdb

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
    # Input:
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    p = p0
    thres = 0.05

    while True:
        X, Y = meshgrid(rect[0]:rect[2], rect[1]:rect[3])
        pdb.set_trace()
        I = RectBivariateSpline(X, Y, It)
        I1 = RectBivariateSpline(X, Y, It1)
        # update
        p = p + dp

        # check whether smaller than thres
        if dp < thres:
            break
    return p
