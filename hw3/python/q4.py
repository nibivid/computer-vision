import numpy as np
from q2 import *
from q3 import *
import skimage.color
from skimage.transform import warp
import pdb

# you may find this useful in removing borders
# from pnc series images (which are RGBA)
# and have boundary regions
def clip_alpha(img):
    img[:,:,3][np.where(img[:,:,3] < 1.0)] = 0.0
    return img

# Q 4.1
# this should be a less hacky version of
# composite image from Q3
# img1 and img2 are RGB-A
# warp order 0 might help
# try warping both images then combining
# feel free to hardcode output size
def imageStitching(img1, img2, H2to1):
    panoImg = None
    # YOUR CODE HERE
    output_shape = (img1.shape[0]+img2.shape[0], img1.shape[0]+img2.shape[0])
    img2to1 = warp(img2, np.linalg.inv(H2to1), output_shape=output_shape)
    panoImg = warp(img1, np.eye(3), output_shape=output_shape)
    panoImg[np.where(panoImg==0)] = img2to1[np.where(panoImg==0)]

    return panoImg

# find H from img2 to img1, using ORB, order: [w, h, 1]
def findH(img1, img2):
    from skimage.feature import ORB,match_descriptors

    # load image
    img1_gray = skimage.color.rgb2gray(img1)
    img2_gray = skimage.color.rgb2gray(img2)

    # extract points
    detector_extractor1 = ORB(n_keypoints=3000)
    detector_extractor1.detect_and_extract(img1_gray)
    detector_extractor2 = ORB(n_keypoints=3000)
    detector_extractor2.detect_and_extract(img2_gray)
    matches = match_descriptors(detector_extractor1.descriptors,
                                detector_extractor2.descriptors)
    match_pts1 = detector_extractor1.keypoints[matches[:,0]].astype(int)
    match_pts2 = detector_extractor2.keypoints[matches[:,1]].astype(int)

    # call RANSAC
    match_pts1 = np.flip(match_pts1, axis=1)
    match_pts2 = np.flip(match_pts2, axis=1)
    H_2to1, _ = computeHransac(match_pts1, match_pts2)
    H_2to1 = H_2to1 / H_2to1[2,2]
    print('tranform H:')
    print(H_2to1)

    return H_2to1


# find H from img2 to img1, using own code, order: [w, h, 1]
def findH_self(img1, img2):
    # load image
    img1_gray = skimage.color.rgb2gray(img1)
    img2_gray = skimage.color.rgb2gray(img2)

    # extract points
    locs1, desc1 = briefLite(img1_gray)
    locs2, desc2 = briefLite(img2_gray)
    matches = briefMatch(desc1, desc2)
    # plotMatches(img1, img2, matches, locs1, locs2)

    match_pts1 = locs1[matches[:,0]]
    match_pts2 = locs2[matches[:,1]]

    # call RANSAC
    match_pts1 = np.flip(match_pts1, axis=1)
    match_pts2 = np.flip(match_pts2, axis=1)
    H_2to1, _ = computeHransac(match_pts1, match_pts2)
    H_2to1 = H_2to1 / H_2to1[2,2]
    print('tranform H:')
    print(H_2to1)

    return H_2to1

# Q 4.2
# you should make the whole image fit in that width
# python may be inv(T) compared to MATLAB
def imageStitching_noClip(img1, img2, H2to1, panoWidth=1280):
    panoImg = None
    # YOUR CODE HERE
    # compute the four corner of the panoImg
    img2_corner = np.array([[0,0,1],[0,img2.shape[0]-1,1],\
                            [img2.shape[1]-1,0,1],[img2.shape[1]-1,img2.shape[0]-1,1]])  # 4 x 3
    img2to1_corner = np.dot(H2to1, img2_corner.T)    # 3 x 4
    img2to1_corner = img2to1_corner / img2to1_corner[2,:]

    img2to1_corner_min = np.min(img2to1_corner, axis=1) # [w, h, 1]
    img2to1_corner_max = np.max(img2to1_corner, axis=1)
    h_min = min(0, img2to1_corner_min[1])
    h_max = max(img1.shape[0]-1, img2to1_corner_max[1])
    w_min = min(0, img2to1_corner_min[0])
    w_max = max(img1.shape[1]-1, img2to1_corner_max[0])
    h_pano = h_max - h_min
    w_pano = w_max - w_min

    # add scale
    scale_factor = panoWidth / w_pano
    panoHeight = int(scale_factor * h_pano)
    M = np.array([[scale_factor, 0, -w_min*scale_factor],
                    [0, scale_factor, -h_min*scale_factor],
                    [0, 0, 1]])       # warp func [w,h]

    # warp
    panoImg = warp(img1, np.linalg.inv(M), output_shape=(panoHeight, panoWidth))
    img2toM = warp(img2, np.linalg.inv(np.dot(M,H2to1)), output_shape=(panoHeight, panoWidth))
    panoImg[np.where(panoImg==0)] = img2toM[np.where(panoImg==0)]
    return panoImg

# Q 4.3
# should return a stitched image
# if x & y get flipped, np.flip(_,1) can help
def generatePanorama(img1, img2):
    panoImage = None
    # YOUR CODE HERE
    bestH2to1 = findH_self(img1, img2)
    # bestH2to1 = findH(img1, img2)
    panoImage = imageStitching_noClip(img1,img2,bestH2to1)

    return panoImage

# Q 4.5
# I found it easier to just write a new function
# pairwise stitching from right to left worked for me!
def generateMultiPanorama(imgs):
    panoImage = None
    # YOUR CODE HERE
    # only two image stich
    if len(imgs) == 2:
        return generatePanorama(imgs[0], imgs[1])

    # left half
    panoImage1 = imgs[len(imgs)//2]
    for i in range(len(imgs)//2-1,-1,-1):
        panoImage1 = generatePanorama(panoImage1,imgs[i])
    # plt.imshow(panoImage1)
    # plt.show()

    # right half
    panoImage2 = imgs[len(imgs)//2]
    for i in range(len(imgs)//2+1,len(imgs)):
        panoImage2 = generatePanorama(panoImage2,imgs[i])
    # plt.imshow(panoImage2)
    # plt.show()

    # two half together
    panoImage = generatePanorama(panoImage1,panoImage2)
    # plt.imshow(panoImage)
    # plt.show()

    return panoImage
