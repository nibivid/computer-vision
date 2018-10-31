import numpy as np
import pdb

# Q3.1
def essentialMatrix(F,K1,K2):
    E = None
    E = np.dot(np.dot(K2.T, F), K1)
    return E

# Q3.2

def triangulate(P1, pts1, P2, pts2):
    P, err = None, None
    N = pts1.shape[0]
    P = np.zeros((N, 4))

    # for each pair of points, find 3D point
    for i in range(N):
        # define A
        A = np.zeros((4, 4))
        A[0, :] = pts1[i, 1] * P1[2, :] - P1[1, :]
        A[1, :] = P1[0, :] - pts1[i, 0] * P1[2, :]
        A[2, :] = pts2[i, 1] * P2[2, :] - P2[1, :]
        A[3, :] = P2[0, :] - pts2[i, 0] * P2[2, :]

        # SVD
        u, s, vh = np.linalg.svd(np.dot(A.T, A))
        P[i, :] = vh[-1, :].T
        P[i, :] = P[i, :] / P[i, -1]

    # pts_hat
    pts1_hat = np.dot(P1, P.T).T
    pts2_hat = np.dot(P2, P.T).T
    for i in range(N):
        pts1_hat[i, :] = pts1_hat[i, :] / pts1_hat[i, -1]
        pts2_hat[i, :] = pts2_hat[i, :] / pts2_hat[i, -1]

    # compute error
    err = np.linalg.norm(pts1 - pts1_hat[:,:2])**2 + np.linalg.norm(pts2 - pts2_hat[:,:2])**2

    # solve
    return P, err
