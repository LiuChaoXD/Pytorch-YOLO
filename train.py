import argparse

from src.trainer import train


def parser_args():
    ROOT = "/home/loch/Project/Pytorch-YOLO"
    parser = argparse.ArgumentParser()
    # ==================================== model settings ===================================
    parser.add_argument('--cfg', type=str, default=ROOT + '/src/config/yolov5l.yaml', help='model.yaml')
    parser.add_argument(
        '--pretrained', type=str, default=ROOT + '/weights/yolov5l.pt', help='initialized/pretrained weights'
    )
    parser.add_argument(
        '--inference', type=str, default=ROOT + '/weights/yolov5l.pt', help='initialized/pretrained weights'
    )
    parser.add_argument('--max_det', type=int, default=300, help='max number of detection')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='iou threshold value for detection')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold value for detection')
    parser.add_argument('--img_size', type=int, default=640, help='images size')

    # ==================================== data settings ====================================
    parser.add_argument('--trainset', type=str, default=ROOT + '/data/trainvalno5k.txt', help='training set path')
    parser.add_argument('--valset', type=str, default=ROOT + '/data/5k.txt', help='validating set path')
    parser.add_argument('--cocos', type=str, default=ROOT + '/data/coco.names', help='coco names')
    parser.add_argument('--n_cpu', type=int, default=8, help='the number of workers for dataloader')
    # ==================================== training settings ====================================

    parser.add_argument('--EPOCH', type=int, default=300, help='Training epoch')
    parser.add_argument('--optimizer', type=str, default="AdamW", help='batch size for training')
    parser.add_argument('--parameter_dir', type=str, default="./parameters", help='path to save parameters')
    parser.add_argument('--checkpoints', type=str, default="./checkpoints", help='path to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--val_interval', type=int, default=1, help='interval for validating in training')
    parser.add_argument('--accumulate_steps', type=int, default=4, help='accumulated gradient update')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='optimizer weight decay 5e-4')

    parser.add_argument('--tensorboard', type=str, default='./tensorboard_log/', help='tensorboard log path')
    parser.add_argument('--continuous', type=str, default='./checkpoints/model.pt', help='continuous training')
    parser.add_argument('--cache', type=bool, default=False, help='use cache to boost data loader')

    parser.add_argument('--amp', type=bool, default=True, help='mixup precision training')
    # ==================================== training settings for loss ====================================
    parser.add_argument('--box', type=float, default=0.05, help='box loss gain')
    parser.add_argument('--cls', type=float, default=0.3, help='cls loss gain')
    parser.add_argument('--cls_pw', type=float, default=1.0, help='cls BCELoss positive_weight')
    parser.add_argument('--obj', type=float, default=0.7, help='obj loss gain (scale with pixels)')
    parser.add_argument('--obj_pw', type=float, default=1.0, help='obj BCELoss positive_weight')
    parser.add_argument('--iou_t', type=float, default=0.20, help='IoU training threshold')
    parser.add_argument('--anchor_t', type=float, default=4.0, help='anchor-multiple threshold')
    parser.add_argument('--fl_gamma', type=float, default=0.0, help='focal loss gamma')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    train(args)
