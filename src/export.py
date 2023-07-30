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

    x = torch.randn((1, 3, img_size, img_size), dtype=torch.float32).to(device)
    for _ in range(2):
        y = model(x)  # dry runs

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
        save_model(trans_model, onnx_path)
        LOGGER.info(f"{colorstr('Half ONNX:')} onnx export correctly!")
    return None


def check_outputs(
    model_cfg: str,
    img_size: int,
    pt_weight: str,
    onnx_path: str,
    device: str,
    half: bool,
):
    # torch.set_printoptions(precision=8)
    input_img = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # load pytorch model
    cfg = check_yaml(model_cfg)
    pt_model = Model(cfg=cfg).eval().to(device)
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
        np.testing.assert_almost_equal(outputs, ort_outputs, decimal=1)
    except Exception as e:
        LOGGER.info(f"{colorstr('ONNX:')} align failure: {e}")
    else:
        LOGGER.info(f"{colorstr('ONNX:')} align the outputs with ONNX and Torch")
