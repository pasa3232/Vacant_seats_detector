#!/bin/bash

for id in 0 1 2 3
do
    python3 yolov5/detect.py \
        --weights yolov5/yolov5s.pt \
        --project runs/detect/simple \
        --name cam$id \
        --source data/simple/cam$id \
        --img 640 --conf 0.25 --save-txt --save-crop 
done
