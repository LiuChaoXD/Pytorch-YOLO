import argparse

from src.export import check_outputs, export_onnx


def parser_args():
    ROOT = "/home/loch/Project/Pytorch-YOLO"
    parser = argparse.ArgumentParser()
    # ==================================== model settings ===================================
    parser.add_argument('--cfg', type=str, default=ROOT + '/src/config/yolov5l.yaml', help='model.yaml')
    parser.add_argument('--weights', type=str, default='./checkpoints/yolo.ckpt', help='trained weights')
    parser.add_argument('--img_size', type=int, default=640, help='images size')
    parser.add_argument('--onnx_path', type=str, default='./yolo.onnx', help='onnx model path')
    parser.add_argument('--simplfy', type=bool, default=True, help='onnxsim simplfy onnx model')
    parser.add_argument('--half', type=bool, default=False, help="export half onnx model, half model only support cuda")
    parser.add_argument('--device', type=str, default='cpu', help="export model type (support cpu and cuda)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    export_onnx(
        model_cfg=args.cfg,
        img_size=args.img_size,
        pt_weight=args.weights,
        onnx_path=args.onnx_path,
        device=args.device,
        simplfy=args.simplfy,
        half=args.half,
    )
    check_outputs(
        model_cfg=args.cfg,
        img_size=args.img_size,
        pt_weight=args.weights,
        onnx_path=args.onnx_path,
        device=args.device,
        half=args.half,
    )
