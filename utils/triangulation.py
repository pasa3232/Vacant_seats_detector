import numpy as np

### triangulation
def triangulation(poses, X, Y):
    K = np.array([
        [975.813843, 0, 960.973816],
        [0, 975.475220, 729.893921],
        [0, 0, 1]
    ])

    constraint_mat = []
    for pose, x, y in zip(poses, X, Y):
        calib = K @ pose
        p1_t = calib[0, :]
        p2_t = calib[1, :]
        p3_t = calib[2, :]
        constraint_mat.append([y * p3_t - p2_t])
        constraint_mat.append([x * p2_t - y * p1_t])
    constraint_mat = np.array(constraint_mat).reshape(1, 8, 4).squeeze()

    # svd for solution
    _, sigma , V = np.linalg.svd(constraint_mat[:6,:])
    vh = V[-1, :]
    p3d = vh[:-1] / vh[-1]

    return p3d
