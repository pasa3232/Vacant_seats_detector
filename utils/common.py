import numpy as np

### triangulation
# poses: list of (4, 4)
def triangulation(poses, X, Y):
    K = np.array([
        [975.813843, 0, 960.973816],
        [0, 975.475220, 729.893921],
        [0, 0, 1]
    ])

    constraint_mat = []
    for pose, x, y in zip(poses, X, Y):
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        pose = np.linalg.inv(pose)
        pose = pose[:3, :]
        calib = K @ pose
        p1_t = calib[0, :]
        p2_t = calib[1, :]
        p3_t = calib[2, :]
        constraint_mat.append([y * p3_t - p2_t])
        constraint_mat.append([x * p2_t - y * p1_t])
        
    constraint_mat = np.array(constraint_mat).reshape(1, 2 * len(poses), 4).squeeze()

    # svd for solution
    _, sigma , V = np.linalg.svd(constraint_mat[:,:])
    vh = V[-1, :]
    # p3d = vh / vh[-1]
    p3d = vh[:-1] / vh[-1]

    return p3d


# backprojects image pixel (a, b) to point on table surface (X, Y, Z)
# Input: n pixels of dimension n x 2, camera pose (R | t) of dimension 3 x 4, plane coefficients (A, B, C, D)
# output: n points on table plane in world coordinate of dimension n x 3 
def pixel2plane(pixels, K, pose, plane_coeffs):
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


### inverse triangulation
# poses: (4, 4)
def reprojection(points_3d, K, pose):
    pose = np.linalg.inv(pose)
    pose = pose[:3, :]
    
    points_3d_h = np.insert(points_3d, points_3d.shape[1], 1, axis=1)
    reprojection = K @ pose @ points_3d_h.T
    points_2d = np.divide(reprojection, reprojection[-1])[:2]

    return points_2d.T


# Change 3d points located on the table plane to layout plane (x, y)
# Input: 3d points np.array(N, 3), plane coeff
# Output: changed points
def plane2layout(points, plane):
    def norm(v):
        return v / np.sqrt(np.sum(v ** 2))
    
    new_z = norm(plane[:3])
    new_x = norm(np.cross(new_z, np.array([0, 1, 0])))
    new_y = norm(np.cross(new_z, new_x))

    def new_p(p):
        return np.array([np.dot(p, new_x), np.dot(p, new_y), np.dot(p, new_z)])

    return np.array([new_p(p)[:-1] for p in points])


# Change layout plane (x, y) to 3d points (x, y, z)
# Input: 2d points np.array(N, 2), plane coeff
# Output: changed points
def layout2plane(points, plane):
    def norm(v):
        return v / np.sqrt(np.sum(v ** 2))
    
    new_z = norm(plane[:3])
    new_x = norm(np.cross(new_z, np.array([0, 1, 0])))
    new_y = norm(np.cross(new_z, new_x))

    res = []
    for x, y in points:
        z = -plane[3] / np.sqrt(np.sum(plane[:3] ** 2))
        res.append(x * new_x + y * new_y + z * new_z)
    return np.array(res)


def p2px(p, mi_px, mx_px, width, height):
    return ((p - mi_px) / (mx_px - mi_px) * np.array([width, height])).astype(int)

def px2p(p, mi_px, mx_px, width, height):
    return p / np.array([width, height]) * (mx_px - mi_px) + mi_px

def get_bbox(points):
    return np.max(points, axis=0), np.min(points, axis=0)



### Homography
# p1 = H * p2
def compute_h(p1, p2):
    # normalization
    # h, w = 2080, 3720
    h, w = 1456, 1928
    p1 = np.array([[p[1] / h, p[0] / w, 1] for p in p1])
    p2 = np.array([[p[1] / h, p[0] / w, 1] for p in p2])

    n, _ = p1.shape
    A = []
    b = [[0] for _ in range(2 * n)]
    for i in range(n):
        x_, y_, _ = p1[i]
        x, y, _ = p2[i]

        A.append([x, y, 1, 0, 0, 0, -x_ * x, -x_ * y, -x_])
        A.append([0, 0, 0, x, y, 1, -y_ * x, -y_ * y, -y_])

    # Add new row to set h_2_2 = 1
    A.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
    b.append([1])

    A = np.array(A)
    b = np.array(b)
    t, _, _, _ = np.linalg.lstsq(A,b, rcond=None)
    H = t[:,0].reshape(3, 3)

    return H

cor_p = [
    [[609, 1025],     [399, 632],    [1491, 819],   [683, 549],     [822, 665],     [1209, 581]],      # cam0
    [[1223, 1082],    [568, 863],    [1523, 798],   [718, 739],     [977, 802],     [1159, 685]],      # cam1
    [[1613, 893],     [834, 897],    [1463, 669],   [775, 755],     [1116, 756],    [1089, 637]],      # cam2
    [[1673, 698],     [1203, 888],   [1297, 510],   [834, 734],     [1179, 653],    [932, 515]]        # cam3
]
