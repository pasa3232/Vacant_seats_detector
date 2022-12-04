import cv2
import numpy as np
import os
import path
import sys
from pathlib import Path
from PIL import Image

from layout import yolo_outputs, yoloToBoxes

# return list of line endpoints
# reference: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
def find_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 9
    h, w = gray.shape
    if (h < 100 or w < 100):
        kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    low_threshold = 10
    high_threshold = 70
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    rho = 1  
    theta = np.pi / 180  
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5  # minimum number of pixels making up a line
    max_line_gap = 40  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),1)

    lines = [lines[i].flatten() for i in range(len(lines))]

    return line_image, lines

# connect nearby lines -> find intersection point -> extend endpoints
MAX = 1000
EPS = 0.01

def slope_intercept(lines):
    # print(f'lines 2, 3:\n{lines[2], lines[3]}')
    slope = [MAX if (lines[i][2]-lines[i][0] < EPS) else (lines[i][3]-lines[i][1]) / (lines[i][2]-lines[i][0]) for i in range(len(lines))]
    y_in = [lines[i][0] if (lines[i][2]-lines[i][0] < EPS) else lines[i][1] - slope[i]*lines[i][0] for i in range(len(lines))]
    return slope, y_in

# assume line_info = [slope, y_in]
def find_intersect(line_info1, line_info2):
    slope1, yin1 = line_info1
    slope2, yin2 = line_info2
    if (slope1 == slope2):
        return MAX, MAX
    elif (slope1 == MAX):
        intersect = yin1, yin2
    elif (slope2 == MAX):
        intersect = yin2, yin1
    else:
        inv_det = 1/(-slope1+slope2)
        intersect = inv_det*(yin1-yin2), inv_det*(yin1*slope2 - yin2*slope1)
    return intersect

slope_threshold = 0.03
parallel_threshold = 5
parallel_high = 10
dist_threshold = 20
# lengthen nearby lines so that they intersect, and merge line segments that are on the same line
# could return 1 or 2 lines
# if this returns 1 line, then this means we have merged two consecutive lines, so we can delete the second line

def make_lines(lines):
    newlines = []
    for i in range(len(lines)):
        x11,y11,x12,y12 = lines[i]
        for j in range(i+1, len(lines), 1):
            x21,y21,x22,y22 = lines[j]
            if (x11 <= x21 and x12 <= x22 and parallel_threshold < np.linalg.norm(np.array([x11, y11] - np.array([x21, y21]))) < parallel_high):
                # print('nearly parallel lines that are apart')
                # create new lines between endpoints
                line3 = np.array([x11,y11,x21,y21])
                line4 = np.array([x12,y12,x22,y22])
                newlines.append(line3)
                newlines.append(line4)
            if (x21 <= x11 and x22 <= x12 and parallel_threshold < np.linalg.norm(np.array([x21, y21] - np.array([x11, y11]))) < parallel_high):
                # print('nearly parallel lines that are apart')
                # create new lines between endpoints
                line3 = np.array([x21,y21,x11,y11])
                line4 = np.array([x22,y22,x12,y12])
                newlines.append(line3)
                newlines.append(line4)
    return newlines


def change_endpoints(line1, line2, slope1, slope2, intersect, w, h):
    x11,y11,x12,y12 = line1
    x21,y21,x22,y22 = line2

    # lines that are close together
    # print(f'slopes: {slope1, slope2}')
    if (abs(slope1 - slope2) < slope_threshold):
        if (x11 <= x21 and x12 <= x22 and np.linalg.norm(np.array([x12, y12] - np.array([x21, y21]))) < parallel_threshold):
            # print('nearly parallel lines that are close together')
            line1[2], line1[3] = x22, y22
            return [line1]
        if (x21 <= x11 and x22 <= x12 and np.linalg.norm(np.array([x11, y11] - np.array([x22, y22]))) < parallel_threshold):
            # print('nearly parallel lines that are close together')
            line1[0], line1[1] = x21, y21
            return [line1]

    # print('lengthening lines')
    if (not (0 <= intersect[0] < w) or not (0 <= intersect[1] < h)):
        return [line1, line2]
    # if the intersection point is not on either line segment, elongate both
    if ((not (x11 <= intersect[0] <= x12) or not (min(y11, y12) <= intersect[1] <= max(y11, y12))) and
        (not (x21 <= intersect[0] <= x22) or not (min(y21, y22) <= intersect[1] <= max(y11, y22)))):
        # print('changed endpoints')
        left_dist1 = np.linalg.norm(np.array(intersect) - np.array([x11, y11]))
        right_dist1 = np.linalg.norm(np.array(intersect) - np.array([x12, y12]))
        left_dist2 = np.linalg.norm(np.array(intersect) - np.array([x21, y21]))
        right_dist2 = np.linalg.norm(np.array(intersect) - np.array([x22, y22]))
        
        if left_dist1 < right_dist1:
            end1, min_dist1 = 1, left_dist1
        else:
            end1, min_dist1 = 3, right_dist1
        
        if left_dist2 < right_dist2:
            end2, min_dist2 = 1, left_dist2
        else:
            end2, min_dist2 = 3, right_dist2

        if (min_dist1 < dist_threshold and min_dist2 < dist_threshold):
            line1[end1-1], line1[end1] = intersect[0], intersect[1]
            line2[end2-1], line2[end2] = intersect[0], intersect[1]
            return [line1, line2, intersect]

    elif (not (x21 <= intersect[0] <= x22) or not (min(y21, y22) <= intersect[1] <= max(y21, y22))):    # if the intersection point is on line1 but not on line2, lengthen line2
        # print('changed endpoints')
        left_dist2 = np.linalg.norm(np.array(intersect) - np.array([x21, y21]))
        right_dist2 = np.linalg.norm(np.array(intersect) - np.array([x22, y22]))
        if left_dist2 < right_dist2:
            end2, min_dist2 = 1, left_dist2
        else:
            end2, min_dist2 = 3, right_dist2
        if (min_dist2 < dist_threshold):
            line2[end2-1], line2[end2] = intersect[0], intersect[1]
            return [line1, line2, intersect]


    elif (not (x11 <= intersect[0] <= x12) or not (min(y11, y12) <= intersect[1] <= max(y11, y12))):    # if the intersection point is on line2 but not on line1, lengthen line1
        # print('changed endpoints')
        left_dist1 = np.linalg.norm(np.array(intersect) - np.array([x11, y11]))
        right_dist1 = np.linalg.norm(np.array(intersect) - np.array([x12, y12]))
        if left_dist1 < right_dist1:
            end1, min_dist1 = 1, left_dist1
        else:
            end1, min_dist1 = 3, right_dist1
        if (min_dist1 < dist_threshold):
            line1[end1-1], line1[end1] = intersect[0], intersect[1]
            return [line1, line2, intersect]

    # if the intersection point is on both line segments, this means the segments are touching, so we do not need to lengthen any of them
    # this part also deals with all other unhandled cases
    return [line1, line2, intersect]
    


def connect_lines(img, lines):
    h, w = img.shape[0], img.shape[1]
    # print(f'h, w: {h, w}')
    # print(f'original lines length: {len(lines)}')
    newlines = make_lines(lines)
    # print(f'new lines length: {len(newlines)}')
    lines += newlines
    # print(f'new lines length: {len(lines)}')
    # print(f'lines:\n{lines}')
    slope, y_in = slope_intercept(lines)
    # print(f'slopes 2, 3:\n{slopes[2], slopes[3]}')
    # print(f'y_in 2, 3:\n{y_in[2], y_in[3]}')

    # print(f'test y for line 2:\n{slopes[2]*lines[2][0]+y_in[2]}')
    # print(f'test y for line 3:\n{slopes[3]*lines[3][0]+y_in[3]}')
    intersections = [[[] for _ in range(len(lines))] for _ in range(len(lines))]

    count = 0
    for i in range(len(lines)):
        x11,y11,x12,y12 = lines[i]
        if (x11 < 0 or y11 < 0 or x12 < 0 or y12 < 0):
            continue
        # if (count < 5): print(f'line i: {lines[i]}')
        for j in range(i+1, len(lines)):
            count += 1
            x21,y21,x22,y22 = lines[j]
            if (x21 < 0 or y21 < 0 or x22 < 0 or y22 < 0):
                continue
            # if (count < 5): print(f'line j: {lines[j]}')
            
            line_info1 = [slope[i], y_in[i]]
            line_info2 = [slope[j], y_in[j]]
            intersect = find_intersect(line_info1, line_info2)
            result = change_endpoints(lines[i], lines[j], slope[i], slope[j], intersect, w, h)
            if (len(result) == 1):
                lines[j] = [-5, -5, -5, -5]
            else:
                lines[i], lines[j] = result[0], result[1]
                if (len(result) == 3):
                    intersections[i][j] = result[2]
                    intersections[j][i] = result[2]
                if (len(result) == 4):
                    lines.append(result[2])
                    lines.append(result[3])
                    

    return lines, slope, y_in, intersections
                
        

def test():
    
    for id in range(4):
        # if ('0000' not in file):
        #     continue
        directory = f'../runs/detect/layout/cam{id}'
        # if ('pic_' not in file):
        #     continue
        # file_num = int(file.split('.')[0])
        # file_num = int((file.split('.')[0]).split('_')[1])
        img_path = directory + '/discretized_tables.jpg'
        img = cv2.imread(img_path)
        img, lines = find_lines(img)
        
        table_bbox = yolo_outputs[f'cam{id}']['table']
        table_bbox = yoloToBoxes(id)
        print(f'table_bbox:\n{table_bbox}')

        bbox_path = directory + '/00000.jpg'
        line_image = cv2.imread(bbox_path)
        
        for line in lines:
            x1,y1,x2,y2 = line
            if (x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0):
                continue
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),1)

        cv2.imwrite(f'result{id}.jpeg', line_image)
        print(f'num lines: {len(lines)}')


if __name__ == '__main__':
    test()


