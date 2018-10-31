import numpy as np
import pdb
from q2 import eightpoint, sevenpoint
from q3 import triangulate
# Q 5.1
# we're going to also return inliers
def ransacF(pts1, pts2, M):
    F = None
    inliers = None

    p = 0.999    # prob p at learst one random is inliers
    e = 0.25     # ratio of outliers
    s = 20      # minimum number of points to fit the model
    N = int(np.log(1-p)/np.log(1-np.power(1-e,s)))    # number of samples
    iterNum = int(2 * N)
    thres = 1.5

    # start RANSAC
    # best
    inlinerNumMax = 0
    locs1 = np.concatenate([pts1, np.ones((pts1.shape[0],1))], axis=1)
    locs2 = np.concatenate([pts2, np.ones((pts2.shape[0],1))], axis=1)

    for i in range(iterNum):
        idx = np.random.randint(locs1.shape[0]-1, size=s)
        locs1_sample = locs1[idx]
        locs2_sample = locs2[idx]
        F_cur = eightpoint(locs1_sample[:,:2], locs2_sample[:,:2], M)
        F_cur = F_cur / F_cur[2,2]
        inliners_cur = []
        for j in range(locs1.shape[0]):
            epipoles = np.dot(locs2, F_cur)
            err = np.abs((epipoles * locs1)/np.reshape(np.linalg.norm(epipoles[:,:2], axis=1),(locs1.shape[0],1)))
            inliners_cur = np.absolute(err) < thres
            inlinerNum = np.sum(inliners_cur)

        if inlinerNum > inlinerNumMax:
            inlinerNumMax = inlinerNum
            # pdb.set_trace()
            inliers = np.nonzero(inliners_cur)
            F = F_cur

    return F, inliers

# Q 5.2
# r is a unit vector scaled by a rotation around that axis
# 3x3 rotatation matrix is R
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# http://www.cs.rpi.edu/~trink/Courses/RobotManipulation/lectures/lecture6.pdf
def rodrigues(r):
    R = None


    return R


# Q 5.2
# rotations about x,y,z is r
# 3x3 rotatation matrix is R
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
def invRodrigues(R):
    r = None

    return r

# Q5.3
# we're using numerical gradients here
# but one of the great things about the above formulation
# is it has a nice form of analytical gradient
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    residuals = None

    return residuals

# we should use scipy.optimize.minimize
# L-BFGS-B is good, feel free to use others and report results
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def bundleAdjustment(K1, M1, p1, K2, M2init, p2,Pinit):
    M2, P = None, None

    return M2,P
