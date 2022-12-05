import os
import sys
import numpy as np
import cv2

from utils.common import *
from pathlib import Path
from shapely.geometry import Polygon


num_cams = 4
cam_poses = {} # key: cami, value: pose
for i in range(num_cams):
    with open(f'./camera_poses/{i:05d}.txt', 'r') as f:
        lines = f.readlines()
        pose = []
        for line in lines:
            data = list(map(float, line.split(" ")))
            pose.append(data)
        pose = np.array(pose)
        cam_poses[f'cam{i}'] = pose.reshape(4, 4)

yolo_outputs = {} # key: cami, value: yolo_output

for i in range(num_cams):

    # outputs of one camera:
    yolo_output = {} # key: class, value: yolo_output
    yolo_output['chair'], yolo_output['table'], yolo_output['person'] = [], [], []

    with open(f'./runs/detect/layout/cam{i}/labels/00000.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = list(map(float, line.split(" ")))

            # check object class
            if data[0] == 56 or data[0] == 11: # chair or bench
                yolo_output['chair'].append(np.array(data[1:5]))
            elif data[0] == 60: # table
                yolo_output['table'].append(np.array(data[1:5]))
            elif data[0] == 0: # person
                yolo_output['person'].append(np.array(data[1:5]))
        yolo_outputs[f'cam{i}'] = yolo_output

#------------------------------Triangluation---------------------------------------

h, w = 1456, 1928
poses = [cam_poses[f'cam{i}'][:3, :] for i in range(num_cams)]
K = np.array([
        [975.813843, 0, 960.973816],
        [0, 975.475220, 729.893921],
        [0, 0, 1]
    ])

triangulated = []
for idx in range(len(cor_p[0])):
    X = [cor_p[i][idx][0] for i in range(num_cams)]
    Y = [cor_p[i][idx][1] for i in range(num_cams)]
    triangulated.append(triangulation(poses, X, Y))
triangulated = np.array(triangulated)
# print(triangulated)

cords = np.hstack((triangulated, np.ones((len(triangulated), 1))))
_, sigma , V = np.linalg.svd(cords)
plane = V[-1, :]
# print(plane)

def distToPlane(point, plane):
    A, B, C, D = plane
    x, y, z = point
    return np.abs(A*x+B*y+C*z+D)/np.sqrt(np.sum(np.power([A, B, C],2)))

# for cord in triangulated:
#     print(distToPlane(cord, plane))

def pixelToCord(pixel, Kinv, M):
    vec = np.hstack(([0], Kinv @ pixel))
    res = np.linalg.inv(M) @ vec
    res = res[:-1]/res[-1]
    return res


#------------------------------Discretization---------------------------------------

# Creates array of bounding box information given yolo-outputs of cam{id}
# Input: id
# Output: an array of [center, bbox] corresponding to a table
#   bboxes  : an array of [center, bbox] corresponding to a table
#   center  : [horizontal, vertical] pixel values
#   bbox    : [[horizontal, vertical] of the 4 corners of the bounding box]
def yoloToBoxes(id):
    bboxes = []
    for i, table in enumerate(yolo_outputs[f'cam{id}']['table']):
        y_, x_, w_, h_ = table
        vertices = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
        vertices = [[x_ + p[0] * h_ / 2, y_ + p[1] * w_ / 2] for p in vertices] # (vertical, horizontal)
        center = np.array([int(y_ * w), int(x_ * h)])   # (horizontal, vertical)

        bbox = []
        for v in vertices:
            point = [int(v[1] * w), int(v[0] * h)] 
            bbox.append(point)
        bbox = np.array(bbox)
        bboxes.append([center, bbox])

    return bboxes

# Weighted distance between two vectors
def distance(p1, p2, weight=[]):
    if len(weight)==0:
        weight = np.ones(len(p1))/len(p1)
    return np.linalg.norm((p1 - p2) * weight)

superPixels = []
for r in [0, 100, 200]:
    for g in [0, 100, 200]:
        for b in [0, 100, 200]:
            superPixels.append([r, g, b])
superPixels = np.array(superPixels)

'''
# Discretize each image into superpixels defined above
for id in range(num_cams):
    img = cv2.imread(f'./data/layout/cam{id}/00000.jpg')
    res = np.zeros(img.shape, dtype=np.uint8) + 255
    w, h, c = img.shape
    for i in range(w):
        for j in range(h):
            data = img[i][j]
            distances = [distance(data, x) for x in superPixels]
            res[i][j] = superPixels[np.argmin(distances)]
    cv2.imwrite(f"./runs/discretize/cam{id}/discretized.jpg", res)
    break
'''


'''
# Draw where pixels are [100, 100, 200] (BGR)
for id in range(num_cams):
    if(id!=1):
        continue
    img = cv2.imread(f'./runs/discretize/cam{id}/discretize_cam{id}.jpg')
    img = (np.round(img/100) * 100).astype(np.uint8)
    res = np.zeros(img.shape, dtype=np.uint8) + 255
    w, h, c = img.shape
    for i in range(w):
        for j in range(h):
            if np.sum(np.abs(img[i][j]-[100, 100, 200]))==0:
                res[i][j] = img[i][j]
    cv2.imwrite(f"./runs/discretize/cam{id}/discretized_tables.jpg", res)
'''

# Search each bounding box and get pixel positions of [100, 100, 200] (BGR)
bboxes_all = [yoloToBoxes(id) for id in range(num_cams)]
pixels_all = []
for id in range(num_cams):
    img = cv2.imread(f'./runs/discretize/cam{id}/discretized.jpg')
    img = (np.round(img/100) * 100).astype(np.uint8)
    w, h, c = img.shape

    pixels = []
    bboxes = bboxes_all[id]
    for i, bbox in enumerate(bboxes):
        minPoints = bbox[-1][0]
        maxPoints = bbox[-1][2]
        data = (np.round(img[minPoints[1]:maxPoints[1], minPoints[0]:maxPoints[0], :]/100) * 100).astype(np.uint8)
        w_, h_, _ = data.shape
        for y in range(h_):
            for x in range(w_):
                if np.sum(np.abs(data[x][y]-[100, 100, 200]))==0:
                    pixels.append([x + minPoints[1], y + minPoints[0]])
    
    pixels_all.append(pixels)


#------------------------------Conversion Example---------------------------------------

Kinv = np.linalg.inv(K)
pose = poses[0]
M = np.vstack((plane, pose))

pixel = cor_p[0][0] + [1]
res = pixelToCord(pixel, Kinv, M)
print("Pixel: ", pixel)
print("Triangulated: ", triangulated[0])
print("PixelToCord of Pixel: ", res)
print("Distance to Plane: ", distToPlane(res, plane))