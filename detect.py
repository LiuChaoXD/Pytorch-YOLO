import argparse
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from src.inference import DetectMultiBackend
from src.utils.logger import LOGGER
from src.utils.metrics import non_max_suppression
from src.utils.plot import Annotator


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        names = fp.read().splitlines()
    names = dict(enumerate(names))
    return names


def preprocess_img(img, device):
    img = img / 255  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img).to(device)
    # img1 = img1.half() if model.fp16 else img1.float()  # uint8 to fp16/32
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    img = img.permute(0, 3, 1, 2)
    return img


def run(args):
    classes = load_classes(args.classes)
    model = DetectMultiBackend(model_cfg=args.cfg, weights=args.weights, device=torch.device('cuda'), fp16=True)
    files = os.listdir(args.data_dir)
    color = (128, 128, 128)
    txt_color = (255, 255, 255)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for img_file in files:
        img0 = cv2.imread(os.path.join(args.data_dir, img_file))
        # img1 for detection
        img0 = cv2.resize(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), dsize=(args.img_size, args.img_size))
        img1 = preprocess_img(img0, device=model.device)
        img1 = img1.half() if model.fp16 else img1.float()  # uint8 to fp16/32

        pred = model(img1)
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, max_det=args.max_det)

        annotator = Annotator(img0, line_width=3)
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = annotator.scale_boxes(img1.shape[2:], det[:, :4], img0.shape).round()
                # Write results
                for *box, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    annotator.box_label(box, label=classes[c])
        annotator.save(os.path.join(args.save_dir, img_file))
        # torch.cuda.empty_cache()


def parser_args():
    ROOT = "/home/loch/Project/Pytorch-YOLO"
    parser = argparse.ArgumentParser()
    # ==================================== model settings ===================================
    parser.add_argument('--cfg', type=str, default=ROOT + '/src/config/yolov5l.yaml', help='model.yaml')
    parser.add_argument('--weights', type=str, default=ROOT + "/checkpoints/yolo.ckpt", help='trained weights')
    parser.add_argument('--img_size', type=int, default=640, help='images size')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='onnx model path')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='onnxsim simplfy onnx model')
    parser.add_argument('--max_det', type=int, default=1000)
    parser.add_argument('--classes', type=str, default=ROOT + '/data/coco.names')
    parser.add_argument('--data_dir', type=str, default=ROOT + '/data/coco128/images/train2017')
    parser.add_argument('--save_dir', type=str, default=ROOT + '/results/')

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    run(args)
