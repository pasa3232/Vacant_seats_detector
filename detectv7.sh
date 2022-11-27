#!/bin/bash

for id in 0 1 2 3
do
    python3 yolov7/detect.py \
        --weights yolov7/yolov7.pt \
        --project runs/detectv7/simple \
        --name cam$id \
        --source data/simple/cam$id \
        --img 640 --conf 0.25 --save-txt
done
