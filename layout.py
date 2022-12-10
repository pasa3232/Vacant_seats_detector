import os
import sys
import argparse
import collections
import time
import numpy as np
import cv2
import random

from tqdm import tqdm
from glob import glob

# from utils.get_chairs import *
from utils.common import *
from pathlib import Path
from itertools import permutations
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

        self.img = [cv2.imread(str(self.data_path / f"cam{id}" / "00000.jpg")) for id in range(num_cams)]
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
        px = []
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
        self.mi_px, self.mx_px = np.array([-1.9008672,  -5.80851849]), np.array([ 2.69350295, -1.42077928])

    def get_objectes(self, cls, filename, source=Path('./runs/detect/simple')):
        output = {obj: {} for obj in cls}
        for id in range(self.num_cams):
            tmp = {obj:[] for obj in cls}
            path = str(source / f"cam{id}" / "labels" / str(filename))
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = list(map(float, line.split(" ")))
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
                    if data[0] in cls:
                        tmp[data[0]].append(bboxes)
            for obj in cls:
                output[obj][f'cam{id}'] = tmp[obj]
        return output
    
    def table_overlap(self,):
        tables = {}
        for id in range(self.num_cams):
            tables[f'cam{id}'] = p2px(self.table_output[f'cam{id}'], self.mi_px, self.mx_px, self.width, self.height)
        edge = {}
        V = []
        checked = {}
        for id in range(self.num_cams):
            for t in range(len(tables[f'cam{id}'])):
                edge[(id, t)] = []
                checked[(id, t)] = False
                V.append((id, t))

        for id1 in range(self.num_cams):
            for id2 in range(id1 + 1, self.num_cams):
                matched = matchByOverlap(tables[f'cam{id1}'], tables[f'cam{id2}'])
                for t1, t2 in matched:
                    edge[(id1, t1)].append((id2, t2))
                    edge[(id2, t2)].append((id1, t1))

        groups = []
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
            groups.append(g)
        return tables, groups
    
    def get_table_layout(self, num_samples = 20):
        bf = cv2.BFMatcher()
        tables, groups = self.table_overlap()

        check = [np.zeros([self.h, self.w]) for _ in range(self.num_cams)]
        points_all = [[] for _ in groups]
        for group in groups:
            inter = None
            for id, tid in group:
                if inter is None:
                    inter = Polygon(tables[f'cam{id}'][tid])
                else:
                    inter = inter.intersection(Polygon(tables[f'cam{id}'][tid]))
            
            samples = []
            while len(samples) < num_samples:
                x = random.randint(int(inter.bounds[0]), int(inter.bounds[2]))
                y = random.randint(int(inter.bounds[1]), int(inter.bounds[3]))
                if inter.contains(Point(x,y)):
                    samples.append(np.array([x, y]))
            
            rgbs = []
            rev = []
            for x, y in samples:
                t1 = px2p(np.array([x, y]), self.mi_px, self.mx_px, width, height).reshape(1, 2)
                t2 = layout2plane(t1, plane_coeffs)
                for id in range(num_cams):
                    r_, c_ = reprojection(t2, K, cam_poses[f'cam{id}'])[0].astype(int)[::-1]
                    rgbs.append(self.img[id][r_][c_])
                    rev.append((id, r_, c_))
            rgbs = np.array(rgbs, dtype=np.float32)
            matches = bf.knnMatch(rgbs, self.vocab, k=1)
            
            myCounter = collections.Counter([m[0].trainIdx for m in matches])
            table_class = max(myCounter, key=myCounter.get)

            pixels = []
            for i, m in enumerate(matches):
                if m[0].trainIdx != table_class:
                    continue
                id, r, c = rev[i]
                if check[id][r][c] != 0:
                    continue

                dx = [0, 1, 0, -1]
                dy = [1, 0, -1, 0]
                queue = [(r, c)]
                check[id][r][c] = 1
                tmp = []
                while len(queue) > 0:
                    x, y = queue.pop()
                    tmp.append([y, x])
                    for x_, y_ in zip(dx, dy):
                        u, v = x + x_, y + y_
                        if self.h > u > 0 and self.w > v > 0 and check[id][u][v] == 0 and bf.knnMatch(np.array([img[id][u][v].astype(np.float32)]), vocab, k=1)[0][0].trainIdx == table_class:
                            queue.append((u, v))
                            check[id][u][v] = 1
                pixels += list(pixel2plane(np.array(tmp), self.K, self.cam_poses[f'cam{id}'][:3,:], self.plane_coeffs))


    def find_overlap(self, output_dict):
        objects_dict = {}
        for id in range(self.num_cams):
            objects_dict[f'cam{id}'] = p2px(output_dict[f'cam{id}'], self.mi_px, self.mx_px, self.width, self.height)

        num_groups = 20
        std_id = 0
        for id in range(self.num_cams):
            if num_groups > len(objects_dict[f'cam{id}']):
                num_groups = len(objects_dict[f'cam{id}'])
                std_id = id

        obj_polygons = [Polygon(bbox) for bbox in objects_dict[f'cam{std_id}']]
        object_groups = [[(std_id, pid)] for pid in range(len(objects_dict[f'cam{std_id}']))]
        for id in range(self.num_cams):
            if id == std_id:
                continue
            l = len(objects_dict[f'cam{id}'])
            poly = [Polygon(bbox) for bbox in objects_dict[f'cam{id}']]
            mx_area = -1.
            res = tuple(range(num_groups))
            for permute in permutations(list(range(l)), num_groups):
                sum_area = 0.
                for j in range(num_groups):
                    sum_area += obj_polygons[j].intersection(poly[permute[j]]).area
                if mx_area < sum_area:
                    mx_area = sum_area
                    res = permute
            for j in range(num_groups):
                obj_polygons[j] = obj_polygons[j].intersection(poly[res[j]])
                object_groups[j].append((id, res[j]))
        mask = np.array([poly.area > 0 for poly in obj_polygons])
        object_groups = np.array(object_groups)

        return objects_dict, object_groups[mask]
    
def find_intersection(objects_dict, groups, option=False):
    group_intersection = []
    for group in groups:
        intersection = None
        if (option == True and len(group) < 4):
            continue
        for cid, pid in group:
            elem = objects_dict[f'cam{cid}'][pid]
            polygon = Polygon(elem)
            if (intersection is None):
                intersection = polygon 
            else:
                intersection = polygon.intersection(intersection)
        group_intersection.append(intersection)
    return group_intersection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cams', type=int, default=4, help='number of cams')
    parser.add_argument('--source', type=str, default='./runs/detect/simple', help='source')
    parser.add_argument('--layout-data', type=str, default='./data/layout', help='layout data directory path')
    parser.add_argument('--layout-detect', type=str, default='./runs/detect/layout', help='layout yolo output directory path')
    opt = parser.parse_args()
    print(opt)

    ## Layout
    print('Layout formulating ...')
    t0 = time.time()
    layout = Layout(data_path=opt.layout_data, detect_path=opt.layout_detect, num_cams=opt.num_cams)
    # layout.get_table_layout()
    print(f'Done. ({time.time() - t0:.3f}s)')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('simple.mp4', fourcc, 2.0, (layout.width, layout.height))
    im = cv2.cvtColor(cv2.imread("layout.png"), cv2.COLOR_BGR2RGB)

    for i in tqdm(range(31)):
        person_output = layout.get_objectes([0], "{:05d}.txt".format(i), source=Path(opt.source))[0]
        people, people_groups = layout.find_overlap(person_output)
        people_intersection = find_intersection(people, people_groups)

        im_clone = im.copy()
        for people in people_intersection:
            x, y = people.exterior.xy
            x, y = int(sum(x) / len(x)), int(sum(y) / len(y))
            cv2.circle(im_clone, (x, y), 30, (150, 100, 50), -1)
        out.write(im_clone)
        # cv2.imwrite(f"runs/{i}.jpg", im_clone)
    out.release()
