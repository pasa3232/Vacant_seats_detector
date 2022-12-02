import os
import sys
import numpy as np
import cv2

from utils.homography import compute_h, cor_p
from utils.triangulation import triangulation
from pathlib import Path
from shapely.geometry import Polygon

### fetch camera poses
num_cams = 4
cam_poses = {} # key: cami, value: pose
for i in range(num_cams):
    with open(f'./params/camera_poses/{i:05d}.txt', 'r') as f:
        lines = f.readlines()
        pose = []
        for line in lines:
            data = list(map(float, line.split(" ")))
            pose.append(data)
        pose = np.array(pose)
        cam_poses[f'cam{i}'] = pose.reshape(4, 4)


### fetch yolo outputs
# output of all cameras
yolo_outputs = {} # key: cami, value: yolo_output

for i in range(num_cams):

    # outputs of one camera:
    yolo_output = {} # key: class, value: yolo_output
    yolo_output['chair'], yolo_output['table'], yolo_output['person'] = [], [], []

    with open(f'./runs/detect/layout/cam{i}/labels/0001.txt', 'r') as f:
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


# ### check yolo output
# img = cv2.imread('./yolo_outputs/cam0_0001.jpg')
# h, w, _ = img.shape
# for coord in yolo_outputs['cam0']['person'][0]:
#     x, y = coord[0], coord[1]
#     cv2.circle(img, (int(w*x), int(h*y)), 5, (0, 255, 0), -1)
# # circle the mean spot
# x_, y_ = np.mean(yolo_outputs['cam0']['person'][0], axis=0)
# cv2.circle(img, (int(w*x_), int(h*y_)), 5, (0, 0, 255), -1)
# cv2.imwrite('person.png', img)

h, w = 2080, 3720

poses = [cam_poses[f'cam{i}'][:3, :] for i in range(num_cams)]

X = [sorted(yolo_outputs[f'cam{i}']['person'], key=lambda x: x[0])[-1][0] * w for i in range(num_cams)]
Y = [sorted(yolo_outputs[f'cam{i}']['person'], key=lambda x: x[0])[-1][1] * h for i in range(num_cams)]

print(f"person 3d point: {triangulation(poses, X, Y)}")

H = []
cam_bbox_pairs = {}

for id in range(num_cams - 1):
    original_img = cv2.imread(f'./runs/detect/layout/cam{id}/0001.jpg')
    img = cv2.imread(f'./runs/detect/layout/cam{id+1}/0001.jpg')
    c_in = cor_p[id]
    c_ref = cor_p[id + 1]
    H.append(compute_h(c_ref, c_in))

    bbox_pairs = []
    for i, table in enumerate(yolo_outputs[f'cam{id}']['table']):
        y_, x_, w_, h_ = table
        vertices = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
        vertices = [[x_ + p[0] * h_ / 2, y_ + p[1] * w_ / 2] for p in vertices] # (vertical, horizontal)
        center = np.array([int(y_ * w), int(x_ * h)])   # (horizontal, vertical)
        newy, newx, newz = H[id] @ np.array([x_, y_, 1]).reshape(3, 1)
        transformed_center = np.array([np.squeeze(int(newx / newz * w)), np.squeeze(int(newy / newz * h))])

        original_bbox = []
        bbox = []

        for v in vertices:
            original_point = [int(v[1] * w), int(v[0] * h)] 
            original_bbox.append(original_point)

            x, y, z = H[id] @ np.array([v[0], v[1], 1]).reshape(3, 1)
            x = x / z * h
            y = y / z * w
            bbox.append([int(y), int(x)])
        
        original_bbox = np.array(original_bbox)
        bbox = np.array(bbox)

        bbox_pairs.append([[center, original_bbox], [transformed_center, bbox]])
        
        original_img = cv2.polylines(original_img, [original_bbox], True, (0, 0, 255), 6)
        img = cv2.polylines(img, [bbox], True, (0, 0, 255), 6)
        box_name = f'Box{i}'

        cv2.circle(original_img, (center[0], center[1]), 3, (0, 0, 255), 6, cv2.LINE_8, 0)
        cv2.circle(img, (transformed_center[0], transformed_center[1]), 3, (0, 0, 255), 6, cv2.LINE_8, 0)
        cv2.putText(original_img, box_name, (original_bbox[0][0], original_bbox[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        cv2.putText(img, box_name, (bbox[0][0], bbox[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    cam_bbox_pairs[f'cam{id}_to_{id+1}'] = bbox_pairs   
        
    save_dir = Path('./runs/homography/simple')
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    cv2.imwrite(f"./runs/homography/layout/cam{id}_to_cam{id+1}_cam{id}.jpg", original_img)
    cv2.imwrite(f"./runs/homography/layout/cam{id}_to_cam{id+1}.jpg", img)



# Computes homography matrix between each pair of neighbouring camers
# H[i]: cam(i) -> H[i] -> cam(i+1)
def readH():
    H_ = []
    for id in range(num_cams - 1):
        c_in = cor_p[id]
        c_ref = cor_p[id + 1]
        H_.append(compute_h(c_ref, c_in))
    return H_


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

# id is camera id
def detectChairs(id):
    """
    Output: 
        list of all pairs between table centers and distances to chairs.
        If the number of tables is M and the number of chairs is N, 
        then this function outputs an MxN list.
    Input: 
        id of camera that specifies the (id)th pictures
    Assumptions:
        We assume that all contents in the list follow the order of the tables and chairs in 
        yolo_outputs[f'cam{id}']
    """
    chairList = []
    for _, table in enumerate(yolo_outputs[f'cam{id}']['table']):
        y_, x_, w_, h_ = table
        vertices = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
        vertices = [[x_ + p[0] * h_ / 2, y_ + p[1] * w_ / 2] for p in vertices] # (vertical, horizontal)
        table_center = np.array([int(y_ * w), int(x_ * h)])   # (horizontal, vertical)
        radius = np.linalg.norm(np.array(vertices[0]) - table_center)
        print(f'radius:\n{radius}')
        
        one_table_distances = []
        for _, chair in enumerate(yolo_outputs[f'cam{id}']['chair']):
            y2, x2, w2, h2 = chair
            vertices2 = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
            vertices2 = [[x2 + p[0] * h2 / 2, y2 + p[1] * w2 / 2] for p in vertices2] # (vertical, horizontal)
            chair_center = np.array([int(y2 * w), int(x2 * h)])   # (horizontal, vertical)
            # print(f'table_center:\n{table_center}\nchair_center:\n{chair_center}')
            distance = np.linalg.norm(table_center - chair_center)
            one_table_distances.append(distance)
        
        one_table_distances = np.array(one_table_distances)
        chairList.append(np.array([table_center, one_table_distances]))
    return chairList


# Transforms the bounding boxes of image from cam{i} with a homography H
# Typically, H is the homography from cam{i} to cam{i+1}
# Input: id, H
# Output: transformed_bboxes
#   transformed_bboxes  : an array of [transformed_center, transformed_bbox] corresponding to a table in cam{i}
#   transformed_center  : [horizontal, vertical] pixel values
#   transformed_bbox    : [[horizontal, vertical] of the 4 corners of the bounding box transformed with H]
def yoloToBoxesTransformed(id, H):
    transformed_bboxes = []
    for i, table in enumerate(yolo_outputs[f'cam{id}']['table']):
        y_, x_, w_, h_ = table
        vertices = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
        vertices = [[x_ + p[0] * h_ / 2, y_ + p[1] * w_ / 2] for p in vertices] # (vertical, horizontal)
        center = np.array([int(y_ * w), int(x_ * h)])   # (horizontal, vertical)
        newy, newx, newz = H @ np.array([x_, y_, 1]).reshape(3, 1)
        transformed_center = np.array([np.squeeze(int(newx / newz * w)), np.squeeze(int(newy / newz * h))])

        transformed_bbox = []
        for v in vertices:
            x, y, z = H @ np.array([v[0], v[1], 1]).reshape(3, 1)
            x = x / z * h
            y = y / z * w
            transformed_bbox.append([int(y), int(x)]) # Transformed bounding box
        transformed_bbox = np.array(transformed_bbox)
        transformed_bboxes.append([transformed_center, transformed_bbox])

    return transformed_bboxes

'''
Here we introduce the notion of "before" and "after" images for convenience and generaltity
A homography H maps an image from "before" to "after" (i.e. before --> H --> after)
In most, if not all, of the usages, before will be cam{i}, after will be cam{i+1} for some index i
'''

# Finds matches between the two given bounding boxes
# Input: after_bboxes, before_bboxes_transformed
#   after_bboxes    : array of bounding box information of "after" image
#   before_bboxes   : array of bounding box information of "before" image transformed with homography to "after"
# Output: selected
#   selected        : an array of indices [i, j]
#   i               : the index of the matched object in "after_bboxes"
#   j               : the index of the matched object in "before_bboxes_transformed"
def matchByOverlap(after_bboxes, before_bboxes_transformed):
    overlaps = []
    for i, after_bbox in enumerate(after_bboxes):
        after_bbox_polygon = Polygon(after_bbox[1])
        for j, before_bbox_transformed in enumerate(before_bboxes_transformed):
            before_bbox_transformed_polygon = Polygon(before_bbox_transformed[1])
            if(after_bbox_polygon.intersects(before_bbox_transformed_polygon)):
                overlap = after_bbox_polygon.intersection(before_bbox_transformed_polygon).area
                total = after_bbox_polygon.area + before_bbox_transformed_polygon.area - overlap
                percentage = overlap/total
                # percentage = overlap/after_bbox_polygon.area
                overlaps.append([i, j, percentage])
    overlaps.sort(key = lambda x : x[-1], reverse=True)
    
    after_indices = set(range(len(after_bboxes)))
    before_indices = set(range(len(before_bboxes_transformed)))

    selected = []
    for i, j, _ in overlaps:
        if i in after_indices and j in before_indices:
            # i : index in bboxes (cam(i+1))
            # j : index in transformed_bboxes (cam(i) -> cam(i+1))
            selected.append([i, j])
            after_indices.remove(i)
            before_indices.remove(j)

    return selected


# Plots the found matches between two images "before" and "after"
# Input: before_idx, after_idx, before_bboxes, after_bboxes, matched
#   before_idx      : index of the "before" image
#   after_idx       : index of the "after" image
#   before_bboxes   : bounding box information of the "before" image
#   after_bboxes    : bounding box information of the "after" image
#   matched         : the indices i,j of the matches between the bounding boxes of tables in "before" and "after" image

# reference: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
def find_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150

    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  
    theta = np.pi / 180  
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
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
    cv2.imshow("pic with lines", line_image)
    cv2.waitKey(0)
# 

def plotMatch(before_idx, after_idx, before_bboxes, after_bboxes, matched):
    before_img = cv2.imread(f'./runs/detect/layout/cam{before_idx}/0001.jpg')
    after_img = cv2.imread(f'./runs/detect/layout/cam{after_idx}/0001.jpg')
    for i, (after, before) in enumerate(matched):
        before_bbox = before_bboxes[before]
        after_bbox = after_bboxes[after]
        cv2.putText(before_img, f'match{i}', (before_bbox[0][0], before_bbox[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(after_img, f'match{i}', (after_bbox[0][0], after_bbox[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    save_dir = Path('./runs/match/layout')
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir    


    cv2.imwrite(f"./runs/match/layout/matched_{before_idx}_{before_idx}_to_{after_idx}.jpg", before_img)
    cv2.imwrite(f"./runs/match/layout/matched_{after_idx}_{before_idx}_to_{after_idx}.jpg", after_img)


H_ = readH()
bboxes_all = [yoloToBoxes(id) for id in range(num_cams)]
transformed_bboxes_all = [yoloToBoxesTransformed(id, H[id]) for id in range(num_cams-1)]

for idx in range(num_cams-1):
    before = idx
    after = idx + 1

    before_bboxes = bboxes_all[before]
    before_bboxes_transformed = transformed_bboxes_all[after-1]
    after_bboxes = bboxes_all[after]
    matched = matchByOverlap(after_bboxes, before_bboxes_transformed)
    plotMatch(before, after, before_bboxes, after_bboxes, matched)

idx = 0
img = cv2.imread(f'data/layout/cam{idx}/0001.jpg')
find_lines(img)