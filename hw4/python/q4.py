import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from q2 import eightpoint
from q3 import essentialMatrix, triangulate
from util import camera2
# Q 4.1
# np.pad may be helpful
def epipolarCorrespondence(im1, im2, F, x1, y1):
    x2, y2 = 0, 0
    
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
    

# Extra credit
def visualizeDense(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2):
    return