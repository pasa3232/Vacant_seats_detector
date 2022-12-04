import numpy as np
import matplotlib.pyplot as plt
import cv2

from camera_models import *

K = np.array([
        [975.813843, 0, 960.973816],
        [0, 975.475220, 729.893921],
        [0, 0, 1]
    ])


# backprojects image pixel (a, b) to point on table surface (X, Y, Z)
# Input: n pixels of dimension n x 2, camera pose (R | t) of dimension 3 x 4, plane coefficients (A, B, C, D)
# output: n points on table surface in world coordinate of dimension n x 3 
def pixel2surface(pixels, pose, plane_coeffs):
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
        points = points[::10]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='y', marker=".")

    plt.tight_layout()
    plt.show()






if __name__ == "__main__":
    # show_world()
    img = cv2.imread('./runs/discretize/cam0/discretized.jpg')
    img = (np.round(img / 100) * 100).astype(np.uint8)
    pixels = np.argwhere(((img[:,:,0] == 100) & (img[:,:,1] == 100) & (img[:,:,2] == 200))) # (b, a)
    pose = np.zeros((3, 4))
    pose[:3, :3] = np.eye(3)
    plane_coeffs = np.array([0.04389121, -0.49583658, -0.25795586, 0.82805701])
    points = pixel2surface(np.flip(pixels, axis=1), pose, plane_coeffs) # change pixels to (a, b)


    show_world(plane_coeffs=plane_coeffs, points=points)
