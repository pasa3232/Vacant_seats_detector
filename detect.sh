#!/bin/bash

for id in 0 1 2 3
do
    python3 yolov7/detect.py \
        --weights yolov7/yolov7.pt \
        --project runs/detect/$1 \
        --name cam$id \
        --source data/$1/cam$id \
        --iou-thres 0.1 \
        --img 640 --conf 0.02 --save-txt --no-trace $2
done
