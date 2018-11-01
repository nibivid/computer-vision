import numpy as np
import scipy
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
            err = np.abs(np.sum(epipoles * locs1, axis=1)/np.linalg.norm(epipoles[:,:2], axis=1))
            inliners_cur = np.absolute(err) < thres
            # pdb.set_trace()
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
    theta = np.linalg.norm(r)
    r = r / theta
    # define matrix
    N = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    R = np.eye(3) + np.sin(theta) * N + (1 - np.cos(theta)) * np.dot(N, N)
    return R


# Q 5.2
# rotations about x,y,z is r
# 3x3 rotatation matrix is R
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
def invRodrigues(R):
    r = None
    A = (R - R.T) / 2
    rho = np.array([A[2,1], A[0,2], A[1,0]])
    rho = rho.T
    s = np.linalg.norm(rho)
    c = (np.trace(R) - 1) / 2
    if s == 0 and c == 1:
        r = np.array([0, 0, 0])
    if s == 0 and c == -1:
        tmp = R + np.eye(3)
        for i in range(3):
            if np.any(tmp[:, i]):
                v = tmp[:, i]
                break
        u = v / np.linalg.norm(v)
        r = np.pi * u
        if np.linalg.norm(r) == np.pi and ((r[0] == 0 and r[1] == 0 and r[2] < 0) \
            or (r[0] == 0 and r[1] < 0) or r[0] < 0):
            r = -r
    else:
        u = rho / s
        theta = np.arctan2(s, c)
        r = u * theta

    return r

# Q5.3
# we're using numerical gradients here
# but one of the great things about the above formulation
# is it has a nice form of analytical gradient
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    residuals = None
    # compute matrix
    # pdb.set_trace()
    P, r2, t2 = x[0], x[1], x[2]
    P = np.reshape(P, (-1, 4))
    t2 = np.reshape(t2, (3, 1))
    R2 = rodrigues(r2)
    M2 = np.concatenate([R2, -np.dot(R2, t2)], axis=1)
    P1 = np.dot(K1, M1)
    P2 = np.dot(K2, M2)

    # compute points
    # pdb.set_trace()
    pts1 = np.dot(P, P1.T)
    pts1[:, 0] = pts1[:, 0] / pts1[:, 2]
    pts1[:, 1] = pts1[:, 1] / pts1[:, 2]
    pts2 = np.dot(P, P2.T)
    pts2[:, 0] = pts2[:, 0] / pts2[:, 2]
    pts2[:, 1] = pts2[:, 1] / pts2[:, 2]

    # diff
    residuals = np.linalg.norm(p1 - pts1[:,:2]) + np.linalg.norm(p2 - pts2[:,:2])

    return residuals

# we should use scipy.optimize.minimize
# L-BFGS-B is good, feel free to use others and report results
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def bundleAdjustment(K1, M1, p1, K2, M2init, p2, Pinit):
    M2, P = None, None

    def residualFunc(vars, K1, M1, p1, K2, p2):
        # optim is flatten of M2init + Pinit
        M2 = np.reshape(vars[:12], (3, 4))
        P = np.reshape(vars[12:], (-1, 4))
        R2 = M2[:, :3]
        r2 = invRodrigues(R2)
        T2 = M2[:,3]
        t2 = -np.dot(np.linalg.inv(R2), T2)
        x =  [P.flatten(), r2, t2]
        return rodriguesResidual(K1, M1, p1, K2, p2, x)

    vars_init = np.concatenate([M2init.flatten(), Pinit.flatten()])
    vars_optim = scipy.optimize.minimize(residualFunc, vars_init, args=(K1, M1, p1, K2, p2))
    M2 = np.reshape(vars_optim['x'][:12], (3, 4))
    P = np.reshape(vars_optim['x'][12:], (-1, 4))

    return M2,P
