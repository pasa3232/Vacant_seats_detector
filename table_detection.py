import os
import sys
import numpy as np
import cv2

from utils.homography import compute_h, cor_p
from utils.triangulation import triangulation
from pathlib import Path
from shapely.geometry import Polygon


### fetch camera poses
num_cams = 4

dir = "./runs/detect/layout/cam1"

