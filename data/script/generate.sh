#!/bin/bash
# Generate the path list for the images
# Example usage: bash script/generate.sh
# parent
# ├── yolov5
# └── datasets
#     └── train.txt  ← downloads here
data_dir='/home/loch/Project/yolov5/dataset/coco128/images/train2017'
save_file='/home/loch/Project/yolov5/dataset/train.txt'
python script/generate.py --data_dir=$data_dir --save_file=$save_file

wait # finish background tasks