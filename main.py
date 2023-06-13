import argparse

from trainer import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ==================================== model settings ===================================
    parser.add_argument('--cfg', type=str, default='./config/yolov5s.yaml', help='model.yaml')
    parser.add_argument('--max_det', type=int, default=300, help='max number of detection')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='iou threshold value for detection')
    parser.add_argument(
        '--conf_thres', type=float, default=0.001, help='confidence threshold value for detection'
    )
    parser.add_argument('--img_size', type=int, default=640, help='images size')

    # ==================================== data settings ====================================
    parser.add_argument(
        '--trainset',
        type=str,
        default='./dataset/coco128/train.txt',
        help='training set path',
    )
    parser.add_argument(
        '--valset',
        type=str,
        default='./dataset/coco128/train.txt',
        help='validating set path',
    )
    parser.add_argument('--cocos', type=str, default='./dataset/coco.names', help='coco names')
    parser.add_argument('--n_cpu', type=int, default=8, help='the number of workers for dataloader')

    # ==================================== training settings ====================================
    parser.add_argument(
        '--pretrained',
        type=str,
        default='./weights/yolov5s.pt',
        help='initialized or pretrained weights',
    )
    parser.add_argument('--EPOCH', type=int, default=300, help='Training epoch')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--val_interval', type=int, default=10, help='interval for validating in training')

    args = parser.parse_args()
    train(args)
