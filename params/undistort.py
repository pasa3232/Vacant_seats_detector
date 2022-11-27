import os
from glob import glob
import cv2
import numpy as np
from multiprocessing import Pool

# camera parameters
color_width = 3840
color_height = 2160
color_K = [
    [1769.60561310104, 0, 1927.08704019384],
    [0, 1763.89532833387, 1064.40054933721],
    [0, 0, 1]
]
color_dist_coeffs = [-0.244052127306437, 0.0597008096110524, 0, 0]

'''
undistorted images are cropped to 3720x2080

K = [
    [1769.60561310104, 0, 1867.08704019384],
    [0, 1763.89532833387, 1024.40054933721],
    [0, 0, 1]
]

1769.6056, 1763.8953, 1867.0870, 1024.4005
'''

# path
data_path = './data/**/**/*.jpg'


def undistort(image_path):
    img_cv = cv2.imread(image_path, -1)
    h, w = img_cv.shape[:2]
    dst = cv2.undistort(img_cv, np.asarray(color_K), np.asarray(color_dist_coeffs))
    cropped = dst[40:h-40, 60:w-60]
    cv2.imwrite(image_path, cropped)


if __name__ == "__main__":
    images = glob(data_path)
    pool = Pool(20)
    pool.map(undistort, images)