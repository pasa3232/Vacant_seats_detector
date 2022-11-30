import numpy as np

### triangulation
def triangulation(poses, X, Y):
    K = np.array([
        [1769.60561310104, 0, 1867.08704019384],
        [0, 1763.89532833387, 1024.40054933721],
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
    _, sigma , V = np.linalg.svd(constraint_mat)
    vh = V[-1, :]
    p3d = vh[:-1] / vh[-1]

    return p3d
