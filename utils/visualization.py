import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import json

from sklearn.cluster import KMeans
from utils.common import *
from utils.camera_models import *
from utils.get_chairs import *

num_cams = 4

# get table points on table plane by backprojecting table pixels of all cameras
# Input: camera poses (R | t) of dimension 4 x 4, plane coefficients (A, B, C, D)
# output: points of table on table plane in world coordinate of dimension n x 3 
# from table.json --> 3d points on plane
def get_table_points(K, poses, plane_coeffs):
    # points for tables backprojected to table plane from all cameras
    
    with open('runs/table.json') as json_file:
        tables = json.load(json_file)["tables"]

    points_all = [[] for i in range(len(tables))]
    for idx, table in enumerate(tables):
        for i in range(num_cams):
            pixels = table[f"cam{i}"]
            pose = poses[f'cam{i}'][:3, :]
            points = pixel2plane(np.flip(pixels, axis=1), K, pose, plane_coeffs)
            points_all[idx] += list(points)

    # for i in range(4):
    #     img = cv2.imread(f'../runs/discretize/cam{i}/discretized.jpg')
    #     img = (np.round(img / 100) * 100).astype(np.uint8)
    #     pixels = np.argwhere(((img[:,:,0] == 100) & (img[:,:,1] == 100) & (img[:,:,2] == 200))) # (b, a)
    #     pose = poses[f'cam{i}'][:3, :]
    #     points = pixel2plane(np.flip(pixels, axis=1), K, pose, plane_coeffs) # change pixels to (a, b)
    #     points_all = points_all + list(points)
    
    return [np.array(points) for points in points_all]



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
def show_world(plane_coeffs=None, points=None, boundaries=None):
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

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
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

    if boundaries is not None:
        color = ['r', 'g', 'b', 'c', 'y', 'm', 'b', 'w']
        for i, boundary in enumerate(boundaries):
            boundary = np.array(boundary)
            ax.scatter(boundary[:, 0], boundary[:, 1], boundary[:, 2], color=color[i])

    plt.tight_layout()
    plt.show()


# printout layout from grouped points
def print_layout(clustered_points, plane_coeffs, mi, mx, width, height):
    # clustered_pixels_layout = []
    # for cluster in clustered_points:
    #     cluster_layout = plane2layout(cluster, plane_coeffs)
    #     pixels_layout = p2px(cluster_layout, mi, mx, width, height)
    #     clustered_pixels_layout.append(cluster_layout)
    
    # layout = np.ones((height, width))
    # for clustered_pixels in clustered_pixels_layout:
    #     layout[clustered_pixels[1], clustered_pixels[0]] = 0

    points = []
    for cluster in clustered_points:
        points += cluster
    
    points_layout = plane2layout(points, plane_coeffs)
    mx, mi = get_bbox(points_layout)
    pixels_layout = p2px(points_layout, mi, mx, width, height)

    layout = np.ones((height, width))
    for pixel in pixels_layout:
        layout[pixel[1], pixel[0]] = 255

    cv2.imwrite('layout.png', layout)


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
        with open(f'./camera_poses/{i:05d}.txt', 'r') as f:
            lines = f.readlines()
            pose = []
            for line in lines:
                data = list(map(float, line.split(" ")))
                pose.append(data)
            pose = np.array(pose)
            cam_poses[f'cam{i}'] = pose.reshape(4, 4)

    # get table points from json
    plane_coeffs = get_plane_coeffs(K, cam_poses)
    points = get_table_points(K, cam_poses, plane_coeffs)

    # get boundary points for each table
    boundaries = []
    for table_cluster in points:
        table_2d = plane2layout(table_cluster, plane_coeffs)
        corners_2d = get_corners_2d(table_2d)
        corners_3d = layout2plane(corners_2d, plane_coeffs)
        boundaries_3d = get_area(corners_3d, m=0.25)
        boundaries.append(boundaries_3d)

    # show all
    show_world(plane_coeffs=plane_coeffs, points=points, boundaries=boundaries)

    # save layout
    # all_points = np.concatenate(points, axis=0)
    # mx, mi = get_bbox(plane2layout(all_points, plane_coeffs))
    # width = 800
    # height = int(width * (mx[0] - mi[0]) / (mx[1] - mi[1]))
    