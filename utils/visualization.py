import numpy as np
import matplotlib.pyplot as plt

from camera_models import *


def show_world():
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


    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)

    A, B, C, D = np.array([0.04389121, -0.49583658, -0.25795586, 0.82805701])
    x, y = np.meshgrid(x, y)
    z = - 1 / C * (A * x + B * y + D)

    surf = ax.plot_surface(x, y, z, alpha=0.2, linewidth=100)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_world()
