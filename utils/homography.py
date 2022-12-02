import numpy as np

### p1 = H * p2
def compute_h(p1, p2):
    # normalization
    # h, w = 2080, 3720
    h, w = ...
    p1 = np.array([[p[0] / h, p[1] / w, 1] for p in p1])
    p2 = np.array([[p[0] / h, p[1] / w, 1] for p in p2])

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

# correspondence points
# cor_p = [
#     [[1116, 671],    [1378, 1680],   [1084, 1998],   [1031, 2045],  [905, 1946],    [1129, 2625]],      # cam0
#     [[993, 1139],    [1533, 613],    [1388, 1835],   [1410, 2239],  [1141, 2620],   [2033, 2380]],      # cam1
#     [[805, 1879],    [909, 1243],    [991, 1613],    [1048, 1720],  [1055, 2474],   [1145, 1178]],      # cam2
#     [[862, 2241],    [891, 1469],    [1038, 1605],   [1120, 1582],  [1320, 2298],   [1112, 1059]]       # cam3
# ]

cor_p = [

]

