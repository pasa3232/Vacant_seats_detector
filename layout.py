import os
import sys
import numpy as np
import cv2


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
                yolo_output['chair'].append(np.array(data[1:3]))
            elif data[0] == 60: # table
                yolo_output['table'].append(np.array(data[1:3]))
            elif data[0] == 0: # person
                yolo_output['person'].append(np.array(data[1:3]))
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


### triangulation for the person
K = np.array([
    [1769.60561310104, 0, 1867.08704019384],
    [0, 1763.89532833387, 1024.40054933721],
    [0, 0, 1]
])
h, w = 2080, 3720
constraint_mat = []
for i in range(num_cams):
    calib = K @ cam_poses[f'cam{i}'][:3, :]
    p1_t = calib[0, :]
    p2_t = calib[1, :]
    p3_t = calib[2, :]

    x_, y_ = sorted(yolo_outputs[f'cam{i}']['person'], key=lambda x: x[0])[-1] # most right person is Taeksoo for all images
    x, y = w * x_, h * y_

    constraint_mat.append([y * p3_t - p2_t])
    constraint_mat.append([x * p2_t - y * p1_t])
constraint_mat = np.array(constraint_mat).reshape(1, 8, 4).squeeze()

# svd for solution
_, sigma , V = np.linalg.svd(constraint_mat)
vh = V[-1, :]
person = vh[:-1] / vh[-1]

print(person)