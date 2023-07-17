import argparse
import sys

sys.path.append("./src")
import cv2
import numpy as np
import onnx
import onnxruntime
import onnxsim
import torch
import torch.nn as nn
from models.yolo import Model

# from onnxmltools.utils import float16_converter
from utils.logger import LOGGER
from utils.utils import check_yaml, colorstr
from onnxmltools.utils import float16_converter
from onnx import load_model, save_model


def export_onnx(
    model_cfg: str,
    img_size: int,
    pt_weight: str,
    onnx_path: str,
    device: str,
    simplfy: bool,
    half: bool,
):
    cfg = check_yaml(model_cfg)  # check YAML
    model = Model(cfg=cfg)
    model.load_state_dict(torch.load(pt_weight))  # load
    model.eval()
    model.to(device)

    x = torch.randn(1, 3, img_size, img_size).to(device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            onnx_path,
            opset_version=18,
            input_names=['input'],
            output_names=['output'],
        )
    model_onnx = onnx.load(onnx_path)  # load onnx model
    try:
        onnx.checker.check_model(model_onnx)
    except Exception as e:
        LOGGER.info(f"{colorstr('ONNX:')} onnx export: {e}")
    else:
        LOGGER.info(f"{colorstr('ONNX:')} onnx export correctly!")

    if simplfy:
        try:
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_path)
        except Exception as e:
            LOGGER.info(f"{colorstr('ONNX:')} simplifier failure: {e}")
    if half:
        onnx_model = load_model(onnx_path)
        trans_model = float16_converter.convert_float_to_float16(
            onnx_model, keep_io_types=True
        )
        save_model(trans_model, "./yolo_half.onnx")
        LOGGER.info(f"{colorstr('Half ONNX:')} onnx export correctly!")
    return None


def check_outputs(model_cfg, img_size, pt_weight, onnx_path, device, half):
    # torch.set_printoptions(precision=8)
    input_img = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # load pytorch model
    cfg = check_yaml(model_cfg)
    pt_model = Model(cfg=cfg).to(device).eval()
    pt_model.load_state_dict(torch.load(pt_weight))  # load
    if half:
        pt_model.half()

    # load onnx model
    providers = (
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if device != 'cpu'
        else ['CPUExecutionProvider']
    )
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    ort_inputs = {'input': input_img}
    ort_outputs = session.run(['output'], ort_inputs)[0]

    if half:
        input_img = torch.from_numpy(input_img).to(device).half()
    else:
        input_img = torch.from_numpy(input_img).to(device)

    outputs = pt_model(input_img)[0]  # pt outputs
    outputs = outputs.detach().cpu().numpy()

    try:
        np.testing.assert_almost_equal(outputs, ort_outputs, decimal=2)
    except Exception as e:
        LOGGER.info(f"{colorstr('ONNX:')} align failure: {e}")
    else:
        LOGGER.info(f"{colorstr('ONNX:')} align the outputs with ONNX and Torch")


# class Ensemble(nn.ModuleList):
#     # Ensemble of models
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, augment=False, profile=False, visualize=False):
#         y = [module(x, augment, profile, visualize)[0] for module in self]
#         # y = torch.stack(y).max(0)[0]  # max ensemble
#         # y = torch.stack(y).mean(0)  # mean ensemble
#         y = torch.cat(y, 1)  # nms ensemble
#         return y, None  # inference, train output


# def fuse_model(pt_model, device=None, inplace=True, fuse=True):
#     # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
#     from models.yolo import Detect, Model

#     model = Ensemble()

#     model.append(pt_model.fuse().eval())  # model in eval mode

#     # Module compatibility updates
#     for m in model.modules():
#         t = type(m)
#         if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
#             m.inplace = inplace  # torch 1.7.0 compatibility
#             if t is Detect and not isinstance(m.anchor_grid, list):
#                 delattr(m, 'anchor_grid')
#                 setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
#         elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
#             m.recompute_scale_factor = None  # torch 1.11.0 compatibility

#     # Return model
#     if len(model) == 1:
#         return model[-1]

#     # Return detection ensemble
#     for k in 'names', 'nc', 'yaml':
#         setattr(model, k, getattr(model[0], k))
#     model.stride = model[
#         torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
#     ].stride  # max stride
#     assert all(
#         model[0].nc == m.nc for m in model
#     ), f'Models have different class counts: {[m.nc for m in model]}'
#     return model


# def export_onnx(
#     model_cfg: str,
#     img_size: int,
#     pt_weight: str,
#     onnx_path: str,
#     half: bool,
#     device: str,
# ):
#     """Export onnx model

#     Args:
#         model_cfg (str): path to model configuration
#         img_size (int): image size
#         pt_weight (str): path to Torch model
#         onnx_path (str): path to onnx model
#         half (bool): half onnx model
#         device (str): cpu or cuda
#     """
#     cfg = check_yaml(model_cfg)  # check YAML
#     model = Model(cfg=cfg).to(device)
#     model.load_state_dict(torch.load(pt_weight))  # load
#     # model = fuse_model(model, device=device)
#     x = torch.randn(1, 3, img_size, img_size)

#     if half and device == 'cuda':  # export half onnx model
#         model.cuda()
#         model.eval()
#         model.half()
#         x = x.half().to(
#             torch.device(device)
#         )  # export half onnx model only supports for GPU
#         print(x)
#     if not half and device == 'cuda':
#         model.cuda()
#         model.fuse()
#         model.eval()
#         x = x.to(torch.device(device))  # export half onnx model only supports for GPU

#     with torch.no_grad():
#         torch.onnx.export(
#             model,
#             x,
#             onnx_path,
#             opset_version=18,
#             input_names=['input'],
#             output_names=['output'],
#         )
#     model_onnx = onnx.load(onnx_path)  # load onnx model
#     try:
#         onnx.checker.check_model(model_onnx)
#     except Exception as e:
#         LOGGER.info(f"{colorstr('ONNX:')} onnx export: {e}")
#     else:
#         LOGGER.info(f"{colorstr('ONNX:')} onnx export correctly!")

#     # if args.simplfy:
#     #     try:
#     #         model_onnx, check = onnxsim.simplify(model_onnx)
#     #         assert check, 'assert check failed'
#     #         onnx.save(model_onnx, args.onnx_path)
#     #     except Exception as e:
#     #         LOGGER.info(f"{colorstr('ONNX:')} simplifier failure: {e}")


# def check_outputs(model_cfg, img_size, pt_weight, onnx_path, half, device):
#     # torch.set_printoptions(precision=8)
#     input_img = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

#     # load pytorch model
#     cfg = check_yaml(model_cfg)
#     pt_model = Model(cfg=cfg).to(device).eval()
#     pt_model.load_state_dict(torch.load(pt_weight))  # load

#     # load onnx model
#     providers = (
#         ['CUDAExecutionProvider', 'CPUExecutionProvider']
#         if device != 'cpu'
#         else ['CPUExecutionProvider']
#     )
#     session = onnxruntime.InferenceSession(onnx_path, providers=providers)
#     ort_inputs = {'input': input_img}
#     ort_outputs = session.run(['output'], ort_inputs)[
#         0
#     ]  # onnxruntime outputs (1, 25200, 85)

#     outputs = pt_model(torch.from_numpy(input_img).to(device))[0]  # pt outputs
#     outputs = outputs.detach().cpu().numpy()

#     try:
#         np.testing.assert_almost_equal(outputs, ort_outputs, decimal=3)
#     except Exception as e:
#         LOGGER.info(f"{colorstr('ONNX:')} align failure: {e}")
#     else:
#         LOGGER.info(f"{colorstr('ONNX:')} align the outputs with ONNX and Torch")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # ==================================== model settings ===================================
#     parser.add_argument(
#         '--cfg', type=str, default='./config/yolov5l.yaml', help='model.yaml'
#     )
#     parser.add_argument(
#         '--weights',
#         type=str,
#         default='./checkpoints/yolo_1.ckpt',
#         help='trained weights',
#     )
#     parser.add_argument('--img_size', type=int, default=640, help='images size')
#     parser.add_argument(
#         '--onnx_path', type=str, default='./yolo.onnx', help='onnx model path'
#     )
#     parser.add_argument(
#         '--simplfy', type=bool, default=True, help='onnxsim simplfy onnx model'
#     )
#     parser.add_argument(
#         '--half',
#         type=bool,
#         default=False,
#         help="export half onnx model, half model only support cuda",
#     )
#     parser.add_argument(
#         '--device',
#         type=str,
#         default='cuda',
#         help="export model type (support cpu and cuda)",
#     )
#     parser.add_argument(
#         '--test_img',
#         type=str,
#         default="./test/images/000000000009.jpg",
#         help='test image for check onnx and pytorch model difference',
#     )

#     args = parser.parse_args()
#     export_onnx(
#         model_cfg=args.cfg,
#         img_size=args.img_size,
#         pt_weight=args.weights,
#         onnx_path=args.onnx_path,
#         half=args.half,
#         device=args.device,
#     )
#     check_outputs(
#         model_cfg=args.cfg,
#         img_size=args.img_size,
#         pt_weight=args.weights,
#         onnx_path=args.onnx_path,
#         half=args.half,
#         device=args.device,
#     )
