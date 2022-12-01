import cv2
import numpy as np
from PIL import Image

# input: p1, p2 are Nx2 matrices consisting of N points in R2
# find H in the equation p1 = H*p2
def compute_h(p1, p2):
    # TODO ...
    N = p1.shape[0]
    q1 = np.ones((N, 3))
    q2 = np.ones((N, 3))
    q1[:, 0:2] = p1
    q2[:, 0:2] = p2
    A = np.zeros((2*N, 9))
    for i in range(N):
        A[2*i, 0:3] = q2[i].T
        A[2*i, 6:] = -1*q1[i, 0]*q2[i].T

        A[2*i+1, 3:6] = q2[i].T
        A[2*i+1, 6:] = -1*q1[i, 1]*q2[i].T

    _, _, VT = np.linalg.svd(A)
    H = VT[-1]
    H = np.reshape(H, (3, 3))
    return H

# point1 = H*point2
# let N1, N2 be normalization matrices for p1 and p2, respectively.
# then N1*point1 = A*(N2*point2), where A is the homography matrix.
# to recover H in point1 = H*point2, re-express the equation above as follows:
# point1 = (inv_N1*A*N2)*point2.
def normalization_matrix(p):
    tx = np.mean(p[:, 0])
    ty = np.mean(p[:, 1])
    center = np.array([tx, ty])
    avg_dist = np.mean(np.linalg.norm((p - center.T), axis=1))
    scale = np.sqrt(2) / avg_dist
    T = np.array(([scale, 0, -1*scale*tx], [0, scale, -1*scale*ty], [0, 0, 1]))
    
    return T

# outputs H that maps p2 to p1
def compute_h_norm(p1, p2):
    # TODO ...
    # find shift and scale for normalization matrix T
    T2 = normalization_matrix(p2)
    T1 = normalization_matrix(p1)
    # convert p1, p2 to homogenous coordinates
    N = p1.shape[0]
    q1 = np.ones((N, 3))
    q2 = np.ones((N, 3))
    q1[:, 0:2] = p1
    q2[:, 0:2] = p2
    # apply normalization matrices to the given sets of coordinates
    q2 = q2@(T2.T)
    q1 = q1@(T1.T)

    A = np.zeros((2*N, 9))
    for i in range(N):
        A[2*i, 0:3] = q2[i].T
        A[2*i, 6:] = -1*q1[i, 0]*q2[i].T

        A[2*i+1, 3:6] = q2[i].T
        A[2*i+1, 6:] = -1*q1[i, 1]*q2[i].T

    _, _, VT = np.linalg.svd(A)
    intermH = VT[-1]
    intermH = np.reshape(intermH, (3, 3))

    invT1 = np.linalg.inv(T1)
    H = (invT1@intermH)@T2
    return H

def bilinear_interpolate(input_img, point):
    maxy, maxx = input_img.shape[0], input_img.shape[1]
    tl_x, tl_y = int(np.floor(point[0])), int(np.floor(point[1])) # top left coordinates
    a, b = (point[0] - tl_x), (point[1] - tl_y)
    tr_x, tr_y = tl_x+1, tl_y
    bl_x, bl_y = tl_x, tl_y+1
    br_x, br_y = tl_x+1, tl_y+1

    # in order of TL, TR, BL, BR
    coordinates = np.array([[tl_x, tl_y], [tr_x, tr_y], [bl_x, bl_y], [br_x, br_y]])
    scales = np.array([(1-a)*(1-b), a*(1-b), (1-a)*b, a*b])
    
    value = 0
    for i, coord in enumerate(coordinates):
        x, y = coord[0], coord[1]
        if (0 <= x < maxx) and (0 <= y < maxy):
            value += input_img[y, x]*scales[i]

    return value

def rectify(igs, p1, p2):
    # TODO ...
    # igs_rec initialize
    maxy, maxx, maxz = igs.shape
    igs_rec = np.zeros((maxy, maxx, maxz))  # dimensions are 3d bc of RGB

    H = compute_h_norm(p2, p1)
    invH = np.linalg.inv(H)
    for y in range(maxy):
        for x in range(maxx):
            point1 =  np.array([x, y, 1])
            point2 = invH@point1
            point2 /= point2[2]     # make sure to divide coordinates by z
            value = bilinear_interpolate(igs, point2)
            igs_rec[y, x] = value

    return igs_rec

def bilinear_interpolate(input_img, point):
    maxy, maxx = input_img.shape[0], input_img.shape[1]
    tl_x, tl_y = int(np.floor(point[0])), int(np.floor(point[1])) # top left coordinates
    a, b = (point[0] - tl_x), (point[1] - tl_y)
    tr_x, tr_y = tl_x+1, tl_y
    bl_x, bl_y = tl_x, tl_y+1
    br_x, br_y = tl_x+1, tl_y+1

    # in order of TL, TR, BL, BR
    coordinates = np.array([[tl_x, tl_y], [tr_x, tr_y], [bl_x, bl_y], [br_x, br_y]])
    scales = np.array([(1-a)*(1-b), a*(1-b), (1-a)*b, a*b])
    
    value = 0
    for i, coord in enumerate(coordinates):
        x, y = coord[0], coord[1]
        if (0 <= x < maxx) and (0 <= y < maxy):
            value += input_img[y, x]*scales[i]

    return value

# H: mapping from input to output
def warp_image(igs_in, igs_ref, H):
    # TODO ...
    maxy, maxx = igs_ref.shape[0], igs_ref.shape[1]
    iny, inx = igs_in.shape[0], igs_in.shape[1]
    igs_warp = np.zeros_like(igs_ref)

    # invH: mapping from output to input
    invH = np.linalg.inv(H)
    for y in range(maxy):
        for x in range(maxx):
            point1 = np.array([x, y, 1])
            point2 = invH@point1
            point2 /= point2[2]
            value = bilinear_interpolate(igs_in, point2)
            igs_warp[y, x] = value

    return igs_warp

def set_cor_rec():
    """
    Output: 
        We return correspondence points between each picture taken by a camera and the destination.
    Algorithm:
        All pictures taken by the same camera will have the same correspondence points.
        Correspondence points are sharp corners and points of large changes in intensity. 
        The input points are points on tables found through OpenCV's SIFT detector, 
        and the output points are found manually.
        All images are warped to an image in bird's eye perspective so that the entrance is on the lower left corner.
    Assumptions:
        We assume that the position of tables do not change in all of the pictures.
        Hence, the correspondence points that we find for four images will be used to warp all images.
    """
    camera_in = []
    # TODO: find points on the upper part of the picture!
# 1) lower right's table: lower left point, 2) low right table upper left point, 3) upper right, 4) low middle table lower right, 5) low middle table upper right
    cam0_in = np.array([[1320, 1546], [1670, 1383], [2448, 1705], [1127, 1432], [1471, 1301]])
    # 2300, 1986
    # orig: 1500, 1752
    cam0_out = np.array([[2300, 1986], [2300, 1686], [3015, 1686], [2076, 1986], [2076, 1686]])
    camera_in.append(cam0_in)
    # img0 = cv2.imread('../data/00/cam0/0001.jpg')
    # img1 = cv2.imread('../data/00/cam1/0001.jpg')
    # img2 = cv2.imread('../data/00/cam2/0001.jpg')
    # img3 = cv2.imread('../data/00/cam3/0001.jpg')
    # images = [img0, img1, img2, img3]

    # i = 0
    # gray0 = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray0, None)
    # pts = cv2.KeyPoint_convert(kp)
    
    # # Marking the keypoint on the image using circles
    # images[i]=cv2.drawKeypoints(gray0,
    #                     kp ,
    #                     images[i],
    #                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # cv2.imwrite('sift-points.jpg', images[i])
    return cam0_in, cam0_out


import os
if __name__ == '__main__':
    print(os.stat('data/00/cam0/0001.jpg'))