import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

from shapely.geometry import Polygon as Poly
from shapely.geometry import Point
from visualization import *
from common import *


def get_corners_3d(table_cluster):
    x_max, _, z_max = np.max(table_cluster, axis=0)
    idx_x_max, _, idx_z_max = np.argmax(table_cluster, axis=0)
    x_min, _, z_min = np.min(table_cluster, axis=0)
    idx_x_min, _, idx_z_min = np.argmin(table_cluster, axis=0)
    
    # four corners: (x_min, y_x_min), (x_max, y_x_max), (x_y_max, y_max), (x_y_min, y_min)
    return np.array([
        np.array([x_min, table_cluster[idx_x_min][1], table_cluster[idx_x_min][2]]), 
        np.array([x_max, table_cluster[idx_x_max][1], table_cluster[idx_x_max][2]]), 
        np.array([table_cluster[idx_z_min][0], table_cluster[idx_z_min][1], z_min]), 
        np.array([table_cluster[idx_z_max][0], table_cluster[idx_z_max][1], z_max])
    ])


def get_corners_2d(table_cluster):
    x_max, y_max = np.max(table_cluster, axis=0)
    x_min, y_min = np.min(table_cluster, axis=0)
    
    # four corners: (x_min, y_x_min), (x_max, y_x_max), (x_y_max, y_max), (x_y_min, y_min)
    return np.array([
        np.array([x_min, y_max]), 
        np.array([x_max, y_max]),
        np.array([x_max, y_min]),
        np.array([x_min, y_min])
    ])



def get_area(points, m):
    count = len(points)
    vectors = np.zeros((count, count, len(points[0])))
    for i in range(count):
        for j in range(count):
            vectors[i][j] = points[i] - points[j]
    norms = [np.linalg.norm(vectors[i][(i+1)%count]) for i in range(count)]
    res = [points[i] + m * (vectors[i][(i+1)%count]/norms[i] + vectors[i][i-1]/norms[i-1]) for i in range(count)]
    return np.array(res)


def get_area_points(points, m):
    count = len(points)
    dim = len(points[0])
    vectors = np.zeros((count, count, dim))
    for i in range(count):
        for j in range(count):
            vectors[i][j] = points[i] - points[j]
    
    pivots = []
    for i in range(count):
        pivots.append(points[i] + vectors[i][i-1] * m/np.linalg.norm(vectors[i][i-1]))
        pivots.append(points[i] + vectors[i][(i+1)%count] * m/np.linalg.norm(vectors[i][(i+1)%count]))
    pivots = np.flip(np.array(pivots).reshape((-1, 2, dim)), axis=1).reshape((-1, 2))
    permutation = np.hstack((np.arange(1, len(pivots)), [0]))
    pivots = pivots[permutation].reshape(-1, 2, dim)

    lines = np.array([pivot[1] - pivot[0] for pivot in pivots])
    norms = [np.linalg.norm(line) for line in lines]
    avg = np.average(norms)

    points = []
    for i, pivot in enumerate(pivots):
        if norms[i]>avg:
            points.append(pivot[0] + lines[i]/4)
            points.append(pivot[0] + lines[i] * 3/4)
        else:
            points.append(pivot[0] + lines[i]/2)
    return np.array(points)

# Creates array of bounding box information given yolo-outputs of cam{id}
# Input: id
# Output: an array of [center, bbox] corresponding to a chair
#   bboxes  : an array of [center, bbox] corresponding to a chair
#   center  : [horizontal, vertical] pixel values
#   bbox    : [[horizontal, vertical] of the 4 corners of the bounding box]
def yoloToBoxesChairs(id):

    h, w = 1456, 1928
    yolo_outputs = {} # key: cami, value: yolo_output
    for i in range(num_cams):
        yolo_output = {} # key: class, value: yolo_output
        yolo_output['chair'] = []
        with open(f'../runs/detect/layout/cam{i}/labels/00000.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = list(map(float, line.split(" ")))
                # check object class
                if data[0] == 56 or data[0] == 11: # chair or bench
                    yolo_output['chair'].append(np.array(data[1:5]))
            yolo_outputs[f'cam{i}'] = yolo_output

    bboxes = []
    for i, table in enumerate(yolo_outputs[f'cam{id}']['chair']):
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


if __name__ == "__main__":
    K = np.array([
        [975.813843, 0, 960.973816],
        [0, 975.475220, 729.893921],
        [0, 0, 1]
    ])


    ### fetch camera poses
    num_cams = 4
    cam_poses = {} # key: cami, value: pose
    for i in range(num_cams):
        with open(f'../camera_poses/{i:05d}.txt', 'r') as f:
            lines = f.readlines()
            pose = []
            for line in lines:
                data = list(map(float, line.split(" ")))
                pose.append(data)
            pose = np.array(pose)
            cam_poses[f'cam{i}'] = pose.reshape(4, 4)

    plane_coeffs = get_plane_coeffs(K, cam_poses)

    clustered_points = get_table_points(K, cam_poses, plane_coeffs)
    clustered_points = [x[::20] for x in clustered_points]

    chairPoints_all = []
    chairPath_all = []

    # get corner points for each table in 3D
    for i in range(num_cams):
        img = cv2.imread(f'../data/layout/cam{i}/00000.jpg')
        chairPoints = []
        chairPaths = []
        for idx, table_cluster in enumerate(clustered_points):
            # Reduce dimension
            table_2d = plane2layout(table_cluster, plane_coeffs)

            # Getting corners in layout --> 3d points
            corners_2d = get_corners_2d(table_2d)
            corners_3d = layout2plane(corners_2d, plane_coeffs)

            # Getting corners of areas
            boundaries_2d = get_area(corners_2d, m=0.20)
            boundaries_3d = layout2plane(boundaries_2d, plane_coeffs)

            # Getting positions to look for chairs
            chairpos_2d = get_area_points(corners_2d, m=0.20)
            chairpos_3d = layout2plane(chairpos_2d, plane_coeffs)

            centers_2d = np.array([corners_2d[0][0] + corners_2d[1][0], corners_2d[0][1] + corners_2d[2][1]])/2
            chairPath_2d = []
            chairPath_3d = []
            step = 10
            for chair_x, chair_y in chairpos_2d:
                chairPath_2d.append(np.array(list(zip(
                    np.linspace(centers_2d[0], chair_x, step+1),
                    np.linspace(centers_2d[1], chair_y, step+1)))))
                chairPath_3d.append(layout2plane(chairPath_2d[-1], plane_coeffs))
                chairPath_2d[-1] = reprojection(chairPath_3d[-1], K, cam_poses[f'cam{i}'])
            chairPath_2d = np.array([x.astype(int) for x in chairPath_2d])

            # Reproject all above back to pixel space
            corners_2d = reprojection(corners_3d, K, cam_poses[f'cam{i}'])
            boundaries_2d = reprojection(boundaries_3d, K, cam_poses[f'cam{i}'])
            chairpos_2d = reprojection(chairpos_3d, K, cam_poses[f'cam{i}'])

            corners_2d = corners_2d.astype(int)
            boundaries_2d = boundaries_2d.astype(int)
            chairpos_2d = chairpos_2d.astype(int)

            chairPoints.append(chairpos_2d)
            chairPaths.append(chairPath_2d)
            for corner, boundary in zip(corners_2d, boundaries_2d):
                cv2.circle(img, list(map(int, corner)), 10, (255, 0, 0), -1)
                # cv2.circle(img, list(map(int, boundary)), 10, (0, 0, 255), -1)
            for chairpos in chairpos_2d:
                cv2.circle(img, list(map(int, chairpos)), 10, (0, 255, 0), -1)
            for chairPath in chairPath_2d:
                for point in chairPath:
                    cv2.circle(img, list(map(int, point)), 5, (0, 255, 255), -1)
            cv2.polylines(img, [corners_2d], isClosed=True, color=(255, 0, 0), thickness=3)
            # cv2.polylines(img, [boundaries_2d], isClosed=True, color=(0, 0, 255), thickness=3)
            cv2.polylines(img, [chairpos_2d], isClosed=True, color=(0, 255, 0), thickness=3)

            cv2.putText(img, str(idx), corners_2d[0] - np.array([20, 20]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=3)
        # cv2.imwrite(f'../runs/get_chairs/cam{i}/corners_00000.jpg', img)
        cv2.imwrite(f'../runs/get_chairs/cam{i}/targets_00000.jpg', img)

        chairPoints_all.append(chairPoints)
        chairPath_all.append(chairPaths)

    bboxes_all = [yoloToBoxesChairs(id) for id in range(num_cams)]

    counts = []

    for idx in range(num_cams):
        pose = cam_poses[f'cam{idx}']
        camera_center = pose[:-1, -1]

        # Targets for each table in the image # shape = (6, 6, 2) : table_count, targets, xy
        chairPoint = np.array(chairPoints_all[idx])
        table_count, targets, _ = chairPoint.shape

        tableCenters = np.array([x[0][0] for x in chairPath_all[idx]])
        
        bboxes = bboxes_all[idx]
        polygons = [(bbox[0], Poly(bbox[1])) for bbox in bboxes]
        occupied = [[0 for i in range(targets)] for j in range(table_count)]
        max = 99999999
        # print(len(polygons))
        for center, polygon in polygons:
            # closestTable = np.argmin([np.linalg.norm(center - tableCenters[i]) for i in range(table_count)])
            # closestChair = np.argmin([np.linalg.norm(center - chairPoint[closestTable][j]) for j in range(targets)])
            center = pixel2plane(np.array([center]), K, pose[:-1, :], plane_coeffs)[0]
            closestTable = -1
            closestChair = -1
            toTableCenter = [np.linalg.norm(center - pixel2plane(np.array([tableCenters[i]]), K, pose[:-1, :], plane_coeffs)[0])
                                            for i in range(table_count)]
            toTarget = [np.linalg.norm(center - pixel2plane(np.array([chairPoint[closestTable][j]]), K, pose[:-1, :], plane_coeffs)[0])
                                            for j in range(targets)]
            while(True):
                closestTable = np.argmin(toTableCenter)
                closestTarget = np.argmin(toTarget)
                if occupied[closestTable][closestTarget] == 1:
                    if toTarget[closestTarget] == max:
                        toTableCenter[closestTable] = max
                    else:
                        toTarget[closestTarget] = max
                else:
                    occupied[closestTable][closestTarget] = 1
                    break
        print(occupied, np.sum(np.array(occupied)), sep = " ")
        counts.append([np.sum(x) for x in occupied])

    counts = np.array(counts)
    # print(counts[:,0])
    for i in range(table_count):
        print("Table", i, ":", np.max(counts[:,i]), "Chairs", sep=" ")



 
