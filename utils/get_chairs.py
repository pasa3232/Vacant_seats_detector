import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
 
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

    # Opening JSON file
    # with open('../runs/table.json') as json_file:
    #     points = json.load(json_file)

    # print(points.keys())
    

    # # # get clustered points
    # points = get_table_points(K, cam_poses, plane_coeffs)
    # print(points.shape)
    # points = remove_outliers(points)
    # points = points[::20] # sample points for less computations
    # bbox_max, bbox_min = get_bbox(points)

    # # Table 3d points by group
    # clustered_points = cluster_tables(points=points, num_tables=6, min_tables=4, max_tables=10)
    # clustered_points[1] = np.delete(clustered_points[1], list(range(3850, 3900)), axis=0)
    # print(np.array(clustered_points[0]).shape)

    clustered_points = get_table_points(K, cam_poses, plane_coeffs)
    clustered_points = [x[::20] for x in clustered_points]


    # get corner points for each table in 3D
    for i in range(4):
        img = cv2.imread(f'../data/layout/cam{i}/00000.jpg')
        for idx, table_cluster in enumerate(clustered_points):
            # Reduce dimension
            table_2d = plane2layout(table_cluster, plane_coeffs)

            # Getting corners in layout
            corners_2d = get_corners_2d(table_2d)

            # Corners back to 3d points
            corners_3d = layout2plane(corners_2d, plane_coeffs)

            # Getting corners of areas
            # boundaries_3d = get_area(corners_3d, m=0.15)
            boundaries_2d = get_area(corners_2d, m=0.15)
            # print(boundaries_2d)
            boundaries_3d = layout2plane(boundaries_2d, plane_coeffs)
            # boundaries_2d = layout2plane(boundaries_2d, plane_coeffs)

            chairpos_2d = get_area_points(corners_2d, m=0.15)
            chairpos_3d = layout2plane(chairpos_2d, plane_coeffs)

            corners_2d = reprojection(corners_3d, K, cam_poses[f'cam{i}'])
            boundaries_2d = reprojection(boundaries_3d, K, cam_poses[f'cam{i}'])
            chairpos_2d = reprojection(chairpos_3d, K, cam_poses[f'cam{i}'])

            corners_2d = corners_2d.astype(int)
            # corners_2d[(1, 2),:] = corners_2d[(2, 1),:]
            boundaries_2d = boundaries_2d.astype(int)
            chairpos_2d = chairpos_2d.astype(int)
            # boundaries_2d[(1, 2),:] = boundaries_2d[(2, 1),:]
            for corner, boundary in zip(corners_2d, boundaries_2d):
                cv2.circle(img, list(map(int, corner)), 10, (255, 0, 0), -1)
                cv2.circle(img, list(map(int, boundary)), 10, (0, 0, 255), -1)
            for chairpos in chairpos_2d:
                cv2.circle(img, list(map(int, chairpos)), 10, (0, 255, 0), -1)
            cv2.polylines(img, [corners_2d], isClosed=True, color=(255, 0, 0), thickness=3)
            cv2.polylines(img, [boundaries_2d], isClosed=True, color=(0, 0, 255), thickness=3)
            cv2.polylines(img, [chairpos_2d], isClosed=True, color=(0, 255, 0), thickness=3)
        # cv2.imwrite(f'../runs/get_chairs/cam{i}/corners_00000.jpg', img)
        cv2.imwrite(f'../runs/get_chairs/cam{i}/targets_00000.jpg', img)
 
