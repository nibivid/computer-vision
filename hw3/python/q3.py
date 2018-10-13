import numpy as np
import skimage.color
import skimage.io
from scipy import linalg
import pdb



# Q 3.1
def computeH(l2, l1):
    H2to1 = np.eye(3)
    # YOUR CODE HERE
    # be careful about the order ot l1 and l2
    A = []
    for i in range(l1.shape[0]):
        A.append([-l1[i,0],-l1[i,1],-1,0,0,0,l1[i,0]*l2[i,0],l1[i,1]*l2[i,0],l2[i,0]])
        A.append([0,0,0,-l1[i,0],-l1[i,1],-1,l1[i,0]*l2[i,1],l1[i,1]*l2[i,1],l2[i,1]])
    A = np.array(A)

    # sometimes get error SVD not converge
    try:
        U,S,V = np.linalg.svd(np.dot(A.T,A))
    except np.linalg.linalg.LinAlgError:
        return None

    H2to1_flatten = V[-1,:]
    H2to1 = np.reshape(H2to1_flatten, [3,3])

    return H2to1

# Q 3.2
def computeHnorm(l1, l2):
    H2to1 = np.eye(3)
    # YOUR CODE HERE
    # translate mean to the origin
    l1_mean = np.mean(l1, axis=0)
    l2_mean = np.mean(l2, axis=0)
    l1_norm = l1 - l1_mean
    l2_norm = l2 - l2_mean

    # scale the largest to sqrt(2)
    l1_max = np.max(np.sqrt(l1_norm[:,0]*l1_norm[:,0]+l1_norm[:,1]*l1_norm[:,1]))
    l2_max = np.max(np.sqrt(l2_norm[:,0]*l2_norm[:,0]+l2_norm[:,1]*l2_norm[:,1]))
    l1_norm = np.sqrt(2) * l1_norm / l1_max
    l2_norm = np.sqrt(2) * l2_norm / l2_max

    # multiply transforms
    H1 = computeH(l1, l1_norm)
    H2 = computeH(l2_norm, l2)
    H2to1_norm = computeH(l1_norm, l2_norm)

    # cases that SVD not converge
    if H1 is None or H2 is None or H2to1_norm is None:
        return None

    H2to1 = np.dot(np.dot(H1, H2to1_norm), H2)

    return H2to1

# Q 3.3
def computeHransac(locs1, locs2):
    bestH2to1, inliers = None, None
    # YOUR CODE HERE
    # define parameters
    p = 0.99    # prob p at learst one random is inliers
    e = 0.2     # ratio of outliers
    s = 4       # minimum number of points to fit the model
    N = int(np.log(1-p)/np.log(1-np.power(1-e,s)))    # number of samples
    delta = 5
    iterNum = int(500 * N)

    # start RANSAC
    inlinerNumMax = 0
    inlinerNum = 0
    bestH2to1 = None
    inliners = None
    locs1 = np.concatenate([locs1, np.ones((locs1.shape[0],1))], axis=1)
    locs2 = np.concatenate([locs2, np.ones((locs2.shape[0],1))], axis=1)
    for i in range(iterNum):
        idx = np.random.randint(locs1.shape[0]-1, size=s)
        locs1_sample = locs1[idx]
        locs2_sample = locs2[idx]
        H2to1 = computeHnorm(locs1_sample[:,:2],locs2_sample[:,:2])

        # meet SVD not converge, try other pts
        if H2to1 is None:
            continue

        locs2to1 = (np.dot(H2to1, locs2.T)).T
        locs2to1 = locs2to1 / locs2to1[:,2:]
        dis = np.sqrt((locs2to1[:,0]-locs1[:,0])*(locs2to1[:,0]-locs1[:,0])+\
                        (locs2to1[:,1]-locs1[:,1])*(locs2to1[:,1]-locs1[:,1]))
        inlinerFlag = dis <= delta
        inlinerNum = np.sum(inlinerFlag)
        # print(inlinerNum)
        if inlinerNum > inlinerNumMax:
            bestH2to1 = H2to1
            inlinerNumMax = inlinerNum
            inliners = inlinerFlag * 1

    return bestH2to1, inliers

# Q3.4
# skimage.transform.warp will help
def compositeH( H2to1, template, img):
    # YOUR CODE HERE
    from skimage.transform import warp
    compositeimg = warp(template, np.linalg.inv(H2to1), output_shape=img.shape[0:2])
    compositeimg[np.where(compositeimg==0)] = img[np.where(compositeimg==0)]
    skimage.io.imshow(compositeimg)
    skimage.io.show()

    return compositeimg


def HarryPotterize():
    # we use ORB descriptors but you can use something else
    from skimage.feature import ORB,match_descriptors
    from skimage.transform import warp, resize
    # YOUR CODE HERE
    # load image
    img_desk = skimage.io.imread('../data/cv_desk.png')
    img_cv = skimage.io.imread('../data/cv_cover.jpg')
    img_hp= skimage.io.imread('../data/hp_cover.jpg')
    img_desk = skimage.color.rgba2rgb(img_desk)
    img_desk_gray = skimage.color.rgb2gray(img_desk)
    img_cv_gray = skimage.color.rgb2gray(img_cv)

    # extract points
    detector_extractor_cv = ORB(n_keypoints=3000)
    detector_extractor_cv.detect_and_extract(img_cv_gray)
    detector_extractor_desk = ORB(n_keypoints=3000)
    detector_extractor_desk.detect_and_extract(img_desk_gray)
    matches = match_descriptors(detector_extractor_cv.descriptors,
                                detector_extractor_desk.descriptors)
    match_pts_cv = detector_extractor_cv.keypoints[matches[:,0]].astype(int)
    match_pts_desk = detector_extractor_desk.keypoints[matches[:,1]].astype(int)

    # call RANSAC
    # strange... why the x and y are inversed...
    match_pts_desk = np.flip(match_pts_desk, axis=1)
    match_pts_cv = np.flip(match_pts_cv, axis=1)
    H_cover2desk, _ = computeHransac(match_pts_desk, match_pts_cv)
    H_cover2desk = H_cover2desk / H_cover2desk[2,2]
    print('tranform H:')
    print(H_cover2desk)
    # cover_warp = warp(resize(img_hp,img_cv.shape), np.linalg.inv(H_cover2desk), output_shape=img_desk.shape[0:2])
    # skimage.io.imshow(cover_warp)
    # skimage.io.show()

    # combine image together
    combined_img = compositeH(H_cover2desk, resize(img_hp,img_cv.shape), img_desk)
    skimage.io.imshow(combined_img)
    skimage.io.show()
    return
