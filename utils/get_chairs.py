import numpy as np
import matplotlib.pyplot as plt
import cv2

from visualization import *


def get_corners_3d(table_cluster):
    x_max, y_max, _ = np.max(table_cluster, axis=1)
    idx_x_max, idx_y_max, _ = np.argmax(table_cluster, axis=1)
    x_min, y_min, _ = np.min(table_cluster, axis=1)
    idx_x_min, idx_y_min, _ = np.argmin(table_cluster, axis=1)
    
    # four corners: (x_min, y_x_min), (x_max, y_x_max), (x_y_max, y_max), (x_y_min, y_min)
    return np.array([
        np.array([x_min, table_cluster[idx_x_min][1], table_cluster[idx_x_min][2]]), 
        np.array([x_max, table_cluster[idx_x_max][1], table_cluster[idx_x_max][2]]), 
        np.array([table_cluster[idx_y_min][0], y_min, table_cluster[idx_y_min][2]]), 
        np.array([table_cluster[idx_y_max][0], y_max, table_cluster[idx_y_max][2]])
    ])


def reprojection(points_3d, pose):
    pose = np.vstack((pose, np.array([0, 0, 0, 1])))
    pose = np.linalg.inv(pose)
    pose = pose[:3, :]
    pass


if __name__ == "__main__":
    K = np.array([
        [975.813843, 0, 960.973816],
        [0, 975.475220, 729.893921],
        [0, 0, 1]
    ])

    plane_coeffs = np.array([0.04389121, -0.49583658, -0.25795586, 0.82805701])

    ### fetch camera poses
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

    # get clustered points
    points = get_table_points(cam_poses, plane_coeffs)
    points = remove_outliers(points)
    points = points[::20] # sample points for less computations
    bbox_max, bbox_min = get_bbox(points)
    clustered_points = cluster_tables(points=points, num_tables=6, min_tables=4, max_tables=10)

    # get corner points for each table in 3D
    for table_cluster in clustered_points:
        corners_3d = get_corners_3d(table_cluster=table_cluster)

