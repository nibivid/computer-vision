import numpy as np
import pdb
from scipy.optimize import fsolve

# Q 2.1
def eightpoint(pts1, pts2, M=1):
    F = None
    # scale
    pts1 = pts1 / M
    pts2 = pts2 / M
    T_rescale = np.array([[1.0/M,0,0],[0,1.0/M,0],[0,0,1.0]])
    # make A
    A = construct_A(pts2, pts1)     # why need to pts2, pts1
    # SVD
    u, s, vh = np.linalg.svd(np.dot(A.T, A))
    F = np.reshape(vh.T[:,-1], (3, 3))
    u_F, s_F, vh_F = np.linalg.svd(F)
    s_F[-1] = 0 # enforce rank 2 on F
    F = np.reshape(np.dot(u_F * s_F, vh_F),(3,3))   # s_F is 3 x 1
    F = np.dot(np.dot(T_rescale.T, F), T_rescale)

    return F

def construct_A(pts1, pts2):
    x = pts1[:,0]
    x_ba = pts2[:,0]
    y = pts1[:,1]
    y_ba = pts2[:,1]

    A = np.ones((pts1.shape[0], 9))     # A is M x 9 matrix
    A[:, 0] = x * x_ba
    A[:, 1] = x * y_ba
    A[:, 2] = x
    A[:, 3] = y * x_ba
    A[:, 4] = y * y_ba
    A[:, 5] = y
    A[:, 6] = x_ba
    A[:, 7] = y_ba

    return A

# Q 2.2
# you'll probably want fsolve
def sevenpoint(pts1, pts2, M=1):
    F = None
    N = pts1.shape[0]
    assert(pts1.shape[0] == 7)
    assert(pts2.shape[0] == 7)

    # rescale
    pts1 = pts1 / M
    pts2 = pts2 / M
    T_rescale = np.array([[1.0/M,0,0],[0,1.0/M,0],[0,0,1.0]])
    # make A
    A = construct_A(pts1, pts2)
    # SVD
    u, s, vh = np.linalg.svd(np.dot(A.T, A))
    F1 = np.reshape(vh.T[:,-1], (3,3))
    F2 = np.reshape(vh.T[:,-2], (3,3))

    def func_F(alpha):
        return np.linalg.det(alpha*F1+(1-alpha)*F2)

    # each for different starting points
    alpha_list = []
    for init in np.arange(-10, 10, 0.01):
        alpha, _, ier, _= fsolve(func_F, x0=init, full_output=1)
        # print(alpha[0])
        alpha = round(alpha[0], 4)
        if alpha not in alpha_list and ier: # find a solution and not repetitive
            alpha_list.append(alpha)
        if len(alpha_list) == 3:
            break
    print(alpha_list)

    F_list = []
    for i in range(len(alpha_list)):
        F = alpha_list[i] * F1 + (1-alpha_list[i]) * F2
        F = np.dot(np.dot(T_rescale.T, F), T_rescale)   # rescale
        F_list.append(F)
    return F_list
