import os
import sys
import numpy as np
import cv2

from utils.homography import compute_h, cor_p
from utils.triangulation import triangulation
from pathlib import Path
from shapely.geometry import Polygon


# reference: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
def find_lines(img, f):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150

    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  
    theta = np.pi / 180  
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 125  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    line_image = np.asarray(line_image, np.float64)
    cv2.imwrite(f, line_image)

### fetch camera poses
num_cams = 4

for id in range(num_cams):
    dir = "./runs/detect/layout/cam{id}/crops/dining table/"
    
    save = 

