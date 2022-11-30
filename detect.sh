#!/bin/bash

for id in 0 1 2 3
do
    python3 yolov7/detect.py \
        --weights yolov7/yolov7.pt \
        --project runs/detect/$1 \
        --name cam$id \
        --source data/$1/cam$id \
        --iou-thres 0.2 \
        --img 640 --conf 0.25 --save-txt --no-trace
done
