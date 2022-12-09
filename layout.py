import os
import sys
import time
import numpy as np
import cv2
import random

from tqdm import tqdm
from glob import glob

from utils.common import *
from pathlib import Path
from shapely.geometry import Polygon, MultiPoint, Point
from matplotlib import pyplot as plt

class Layout:
    def __init__(self, data_path="./data/layout", detect_path="./runs/detect/layout", num_cams=4, width=800):
        self.data_path = Path(data_path)
        self.detect_path = Path(detect_path)
        self.h, self.w = 1456, 1928

        self.K = np.array([
            [975.813843, 0, 960.973816],
            [0, 975.475220, 729.893921],
            [0, 0, 1]
        ])
        self.num_cams = num_cams

        self.cam_poses = self.get_poses()
        self.plane_coeffs = get_plane_coeffs(self.K, self.cam_poses)

        self.img = [cv2.imread(str(data_path / f"cam{id}" / "00000.jpg")) for id in range(num_cams)]
        self.vocab, self.vocab_col = self.cluster_tables()
        self.get_tables()

        self.width = width
        self.height = int(width * (self.mx_px[0] - self.mi_px[0]) / (self.mx_px[1] - self.mi_px[1]))


    def get_poses(self,):
        poses = {}
        for i in range(self.num_cams):
            with open(f'./camera_poses/{i:05d}.txt', 'r') as f:
                lines = f.readlines()
                pose = []
                for line in lines:
                    data = list(map(float, line.split(" ")))
                    pose.append(data)
                pose = np.array(pose)
                poses[f'cam{i}'] = pose.reshape(4, 4)
        return poses

    def cluster_tables(self, step=5):
        rgbs = []
        for id in range(self.num_cams):
            im = self.img[id]
            for x in range(0, self.w, step):
                for y in range(0, self.h, step):
                    rgbs.append(im[y][x].tolist())
        rgbs = np.array(rgbs, dtype=np.float32)

        return kmeans(rgbs)

    def get_tables(self,):
        self.table_output = {}
        for id in range(self.num_cams):
            tables = []
            with open(str(self.detect_path / f"cam{id}" / "labels" / "00000.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = list(map(float, line.split(" ")))
                    if data[0] == 60: # table
                        bboxes = np.array([
                            [(data[1] - data[3] / 2) * self.w, (data[2] - data[4] / 2) * self.h],
                            [(data[1] - data[3] / 2) * self.w, (data[2] + data[4] / 2) * self.h],
                            [(data[1] + data[3] / 2) * self.w, (data[2] + data[4] / 2) * self.h],
                            [(data[1] + data[3] / 2) * self.w, (data[2] - data[4] / 2) * self.h],
                        ])
                        # cam plane -> 3d position
                        bboxes = pixel2plane(bboxes, self.K, self.cam_poses[f'cam{id}'][:3,:], self.plane_coeffs)

                        # 3d position -> table plane 2d
                        bboxes = plane2layout(bboxes, self.plane_coeffs)
                        tables.append(bboxes)
                mx, mi = get_bbox(np.concatenate(tables, axis=0))
                px += [mx] + [mi]
                self.table_output[f'cam{id}'] = tables
        self.mx_px, self.mi_px = get_bbox(np.stack(px, axis=0))
    
    def find_overlap(self, output_dict):
        plt.figure(figsize=(16, 16))
        out = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        objects_dict = {}
        for id in range(self.num_cams):
            objects_dict[f'cam{id}'] = p2px(output_dict[f'cam{id}'], self.mi_px, self.mx_px, self.width, self.height)
            out = cv2.polylines(out, objects_dict[f'cam{id}'], True, col[id], 2)

        plt.subplot(3, 3, 1)
        plt.imshow(out)

        edge = {}
        V = []
        checked = {}
        for id in range(self.num_cams):
            for t in range(len(objects_dict[f'cam{id}'])):
                edge[(id, t)] = []
                checked[(id, t)] = False
                V.append((id, t))

        for id1 in range(self.num_cams):
            for id2 in range(id1 + 1, self.num_cams):
                matched = matchByOverlap(objects_dict[f'cam{id1}'], objects_dict[f'cam{id2}'])
                for t1, t2 in matched:
                    edge[(id1, t1)].append((id2, t2))
                    edge[(id2, t2)].append((id1, t1))

        object_groups = []
        for v in V:
            if checked[v]:
                continue
            checked[v] = True
            queue = [v]
            g = []
            while len(queue) > 0:
                now = queue.pop(0)
                g.append(now)
                for u in edge[now]:
                    if not checked[u]:
                        checked[u] = True
                        queue.append(u)
            object_groups.append(g)
        
        return objects_dict, object_groups
    
    def find_intersection(objects_dict, groups, option):
        group_intersection = []
        for i, group in enumerate(groups):
            intersection = None
            if (option == True and len(group) < 4):
                continue
            for cid, pid in group:
                elem = objects_dict[f'cam{cid}'][pid]
                polygon = Polygon(elem)
                print(f'polygon: {polygon}')
                if (intersection is None):
                    intersection = polygon 
                else:
                    intersection = polygon.intersection(intersection)
            group_intersection.append(intersection)
        return group_intersection 