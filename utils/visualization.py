import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from camera_models import *


# backprojects image pixel (a, b) to point on table surface (X, Y, Z)
# Input: n pixels of dimension n x 2, camera pose (R | t) of dimension 3 x 4, plane coefficients (A, B, C, D)
# output: n points on table plane in world coordinate of dimension n x 3 
def pixel2plane(pixels, pose, plane_coeffs):
    pixels, pose, plane_coeffs = np.array(pixels), np.array(pose), np.array(plane_coeffs)

    pose = np.vstack((pose, np.array([0, 0, 0, 1])))
    pose = np.linalg.inv(pose)
    pose = pose[:3, :]

    pixels_h = np.insert(pixels, pixels.shape[1], 1, axis=1)
    back_projection = np.linalg.inv(K) @ pixels_h.T
    b = np.vstack((np.zeros(pixels.shape[0]), back_projection))

    A = np.vstack((plane_coeffs, pose))

    points = np.linalg.inv(A) @ b
    points = np.divide(points, points[-1])[:3]
    return points.T


# get table points on table plane by backprojecting table pixels of all cameras
# Input: camera poses (R | t) of dimension 4 x 4, plane coefficients (A, B, C, D)
# output: points of table on table plane in world coordinate of dimension n x 3 
def get_table_points(poses, plane_coeffs):
    # points for tables backprojected to table plane from all cameras
    points_all = []

    for i in range(4):
        img = cv2.imread(f'./runs/discretize/cam{i}/discretized.jpg')
        img = (np.round(img / 100) * 100).astype(np.uint8)
        pixels = np.argwhere(((img[:,:,0] == 100) & (img[:,:,1] == 100) & (img[:,:,2] == 200))) # (b, a)
        pose = poses[f'cam{i}'][:3, :]
        points = pixel2plane(np.flip(pixels, axis=1), pose, plane_coeffs) # change pixels to (a, b)
        points_all = points_all + list(points)
    
    return np.array(points_all)



# remove outlier points
# Input: all points
# Output: all points with outliers removed
def remove_outliers(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.015)

    return np.array(pcd.points)


# cluster table points
# Input: all table points on table surface
# Output: clusterd table points
def cluster_tables(points, num_tables, min_tables, max_tables):
    # range_n_clusters = range(min_tables, max_tables + 1)
    # scores = [0, 0, 0, 0]
    # for n_clusters in range_n_clusters:
    #     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    #     cluster_labels = clusterer.fit_predict(points)
    #     silhouette_avg = silhouette_score(points, cluster_labels)
    #     scores.append(silhouette_avg)
    #     print(
    #         "For n_clusters =",
    #         n_clusters,
    #         "The average silhouette_score is :",
    #         silhouette_avg,
    #     )
    clusterer = KMeans(n_clusters=num_tables, random_state=10)
    cluster_labels = clusterer.fit_predict(points)
    grouped_tables = [[] for _ in range(num_tables)]
    for idx, label in enumerate(cluster_labels):
        grouped_tables[label].append(points[idx])

    return grouped_tables


# https://github.com/mnslarcher/camera-models/blob/main/camera-models.ipynb
def show_world(plane_coeffs=None, points=None):
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


    # world = cam0
    world_origin = np.zeros(3)
    dx, dy, dz = np.eye(3)
    world_frame = ReferenceFrame(
        origin=world_origin, 
        dx=dx, 
        dy=dy,
        dz=dz,
        name="World=cam0",
    )

    R = np.linalg.inv(cam_poses['cam1'])[:3, :3]
    dx, dy, dz = R
    t = np.linalg.inv(cam_poses['cam1'])[:3, 3]
    t = -R.T @ t
    camera_frame_1 = ReferenceFrame(
        origin=t,
        dx=dx, 
        dy=dy,
        dz=dz,
        name="cam1",
    )

    R = np.linalg.inv(cam_poses['cam2'])[:3, :3]
    dx, dy, dz = R
    t = np.linalg.inv(cam_poses['cam2'])[:3, 3]
    t = -R.T @ t
    camera_frame_2 = ReferenceFrame(
        origin=t,
        dx=dx, 
        dy=dy,
        dz=dz,
        name="cam2",
    )

    R = np.linalg.inv(cam_poses['cam3'])[:3, :3]
    dx, dy, dz = R
    t = np.linalg.inv(cam_poses['cam3'])[:3, 3]
    t = -R.T @ t
    camera_frame_3 = ReferenceFrame(
        origin=t,
        dx=dx, 
        dy=dy,
        dz=dz,
        name="cam3",
    )

    ax = plt.axes(projection="3d")
    world_frame.draw3d()
    camera_frame_1.draw3d()
    camera_frame_2.draw3d()
    camera_frame_3.draw3d()
    set_xyzlim3d(-3, 3)
    ax.set_title(f"World")

    if plane_coeffs is not None:
        x = np.linspace(-4, 4, 10)
        y = np.linspace(-4, 4, 10)

        A, B, C, D = plane_coeffs
        x, y = np.meshgrid(x, y)
        z = - 1 / C * (A * x + B * y + D)

        surf = ax.plot_surface(x, y, z, alpha=0.2, linewidth=100)

    if points is not None:
        color = ['r', 'g', 'b', 'c', 'y', 'm', 'b', 'w']
        for i, group in enumerate(points):
            group = np.array(group)
            ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=color[i], marker=".")

    plt.tight_layout()
    plt.show()


def get_bbox(points):
    return np.max(points, axis=0), np.min(points, axis=0)


# printout layout from grouped points
def print_layout(points, bbox_max, bbox_min, size):

    bbox_size = bbox_max - bbox_min

    layout = 255 * np.ones(size)
    for group in points:
        group = np.array(group)
        group -= bbox_min
        group /= bbox_size.max()
        group *= size[0]
        
        for point in group:
            point = list(map(int, point))
            layout[point[2], point[0]] = 0
    
    cv2.imwrite('layout.png', layout)


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

    points = get_table_points(cam_poses, plane_coeffs)
    points = remove_outliers(points)
    bbox_max, bbox_min = get_bbox(points)
    clustered_points = cluster_tables(points=points[::20], num_tables=6, min_tables=4, max_tables=10)
    show_world(plane_coeffs=plane_coeffs, points=clustered_points)
    print_layout(clustered_points, bbox_max, bbox_min, size=(400, 400))