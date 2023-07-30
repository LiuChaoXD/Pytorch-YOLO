import argparse

from src.validater import evaluation


def parser_args():
    ROOT = "/home/loch/Project/Pytorch-YOLO"
    parser = argparse.ArgumentParser()
    # ==================================== model settings ===================================
    parser.add_argument('--cfg', type=str, default=ROOT + '/src/config/yolov5l.yaml', help='model.yaml')

    parser.add_argument(
        '--inference', type=str, default=ROOT + '/weights/yolov5l.pt', help='initialized/pretrained weights'
    )
    parser.add_argument('--max_det', type=int, default=300, help='max number of detection')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='iou threshold value for detection')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold value for detection')
    parser.add_argument('--img_size', type=int, default=640, help='images size')

    # ==================================== data settings ====================================
    parser.add_argument('--valset', type=str, default=ROOT + '/data/5k.txt', help='validating set path')
    parser.add_argument('--cocos', type=str, default=ROOT + '/data/coco.names', help='coco names')
    parser.add_argument('--n_cpu', type=int, default=8, help='the number of workers for dataloader')
    # ==================================== training settings ====================================
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--cache', type=bool, default=False, help='use cache to boost data loader')
    parser.add_argument('--amp', type=bool, default=True, help='mixup precision training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    evaluation(args)
