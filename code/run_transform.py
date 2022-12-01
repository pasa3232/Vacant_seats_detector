import numpy as np
import os
import rectify
import time
from PIL import Image
from scipy.interpolate import RectBivariateSpline


def set_cor_pts():
    """
    Input: 
        none
    Output: 
        the 6 pairs of correspondence points for images i, i+1, for all i in range(3)
        points are
    Algorithm:
        as the points are found manually, no algorithm is used.
    Assumptions: 
        We assume that the correspondence points are associated with the following files, respectively:
        "data/layout/cam{i}/0001.jpg" for i in range(4).
        It is not the case that the same world coordinates are used for these pairs of points.
    """
    # cam0: lower right table's lower left pt, upper right pt, upper 2nd to leftmost chair, long table rightmost chair
    points = [[] for _ in range(3)]
    set0_cam0 = np.array([[1327, 1550], [1680, 1384], [2159, 879], [2689, 1031]])
    set0_cam1 = np.array([[132, 1520], [616, 1532], [3010, 1191], [3011, 1962]])
    points[0] = np.array([set0_cam0, set0_cam1])

    set1_cam1 = np.array([[167, 1255], [414, 1166], [1760, 1032], [2240, 1055], [3016, 1191], [2968, 1591]])
    set1_cam2 = np.array([[816, 864], [959, 850], [2120, 877], [2394, 955], [2525, 1119], [2502, 1510]])
    points[1] = np.array([set1_cam1, set1_cam2])

    set2_cam2 = np.array([[563, 1617], [578, 1484], [816, 865], [1194, 879], [2305, 1402], [2503, 1511]])
    set2_cam3 = np.array([[631, 1364], [744, 1268], [1207, 790], [1489, 847], [2010, 1651], [2050, 1844]])
    points[2] = np.array([set2_cam2, set2_cam3])
    
    return points

# H: mapping from input to output
def homography():
    points = set_cor_pts()
    H = []
    for i in range(3):
        temp = rectify.compute_h_norm(points[i][1], points[i][0])   
        H.append(temp)
    return H

def test(images):
    """
    Check if homography() works by warping each input image to the perspective of the output image
    """
    start = time.time()
    maxy, maxx = images[0].shape
    output = []
    for _ in range(4):
        output.append(np.zeros((maxy, maxx)))
    output = np.array(output)
    
    H = homography()
    # invH: mapping from output to input
    invH = [np.linalg.inv(H[i]) for i in range(len(H))]
    print(f'invH length and shape:\n{len(invH), invH[0].shape}')

    img_start = np.array([0, 0, 1])
    img_end = np.array([maxx-1, maxy-1, 1])

    out_images = []

    for i in range(3):
        start_out = invH[i] @ img_start
        warped_start_x, warped_start_y = start_out[0] / start_out[2], start_out[1] / start_out[2]
        end_out = invH[i] @ img_end
        warped_end_x, warped_end_y = end_out[0] / end_out[2], end_out[1] / end_out[2]
        warped_x_grid = np.linspace(warped_start_x, warped_end_x, maxx)
        warped_y_grid = np.linspace(warped_start_y, warped_end_y, maxy)
        warped_x_mesh, warped_y_mesh = np.meshgrid(warped_x_grid, warped_y_grid)
        out_x = np.linspace(0, maxx-1, maxx)
        out_y = np.linspace(0, maxy-1, maxy)
        spline = RectBivariateSpline(out_x, out_y, images[i+1].T)
        warped_img = spline.ev(warped_x_mesh, warped_y_mesh)
        out_images.append(warped_img)
        end = time.time()
        print(f'time for image {i}: {end-start} seconds')
        start = end

    # for i in range(3):
    #     warped_img = rectify.warp_image(images[i], images[i+1], H[i])
    #     out_images.append(warped_img)
    #     end = time.time()
    #     print(f'time for image {i}: {end-start} seconds')
    #     start = end

    return np.array(out_images)


def transform(bbox1, bbox2, H):
    """
    Input: 
        The inputs to this function are lists of bounding box data for two images taken at the same time from adjacent cameras.
        points on cam0 and cam1 images, on cam1 and cam2 images, on cam2 and cam3 images
    Output: 
        The output is a list of pairs of input points and their transformed coordinates
    Algorithm: 
        We find 6 pairs of correspondence points by hand in order to compute the transformation matrix W.
        Then for each center point in bboxes1, we find its predicted correspondence point through W.
        We store the center point and the predicted point as a mapped pair into the output dictionary.
    Assumptions:
        We assume that the inputted bounding box data are those of pictures in the 'layout' directory.
    """
    output = []
    for i in range(len(bbox1)):
        coord = H @ bbox1[i]
        output.append({f'{bbox1[i]}': coord})
    
    # do some kind of comparison between outputs and bbox2
    return output


def visualize():
    """
    Visualize the outputs of 
    """
    ...

if __name__ == '__main__':
    
    images = []
    for i in range(4):
        images.append(np.array(Image.open(f'../data/layout/cam{i}/0001.jpg').convert('L')))

    outputs = test(images)
    output_images = [Image.fromarray((outputs[i]).astype(np.uint8)) for i in range(len(outputs))]
    
    for i in range(len(output_images)):
        output_images[i].save(f'{i:05d}.jpg')

