import os
import sys
import numpy as np
import cv2

from utils.homography import compute_h, cor_p
from utils.triangulation import triangulation
from pathlib import Path

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
        center = np.array([int(x_ * h), int(y_ * w)])
        newx, newy, newz = H[id] @ np.array([y_, x_, 1]).reshape(3, 1)
        transformed_center = np.array([np.squeeze(int(newy / newz * h)), np.squeeze(int(newx / newz * h))])

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

        # cv2.circle(original_img, (center[0], center[1]), 3, (0, 0, 255), 6, cv2.LINE_8, 0)
        # cv2.circle(img, (transformed_center[0], transformed_center[1]), 3, (0, 0, 255), 6, cv2.LINE_8, 0)
        cv2.putText(original_img, box_name, (original_bbox[0][0], original_bbox[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        cv2.putText(img, box_name, (bbox[0][0], bbox[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    cam_bbox_pairs[f'cam{id}_to_{id+1}'] = bbox_pairs   
        
    save_dir = Path('./runs/homography/simple')
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    cv2.imwrite(f"./runs/homography/layout/cam{id}_to_cam{id+1}_cam{id}.jpg", original_img)
    cv2.imwrite(f"./runs/homography/layout/cam{id}_to_cam{id+1}.jpg", img)
