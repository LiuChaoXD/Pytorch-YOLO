import sys
from pathlib import Path

sys.path.append("./src")
import numpy as np
import torch
import torch.nn as nn
from utils.logger import LOGGER
from utils.utils import check_yaml, colorstr


class DetectMultiBackend(nn.Module):
    def __init__(
        self,
        model_cfg='path/to/yolo/config',
        weights="yolov5s.pt",
        device=torch.device('cpu'),
        fp16=False,
    ):
        super().__init__()
        self.pt, self.onnx = self._model_type(weights)
        fp16 &= self.pt or self.onnx
        self.fp16 = fp16
        self.stride = 32  # default stride
        self.device = device
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if self.pt:  # PyTorch
            from models.yolo import Model

            cfg = check_yaml(model_cfg)  # check YAML
            model = Model(cfg=cfg)
            model.load_state_dict(torch.load(weights))  # load
            model.eval().to(device)
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        elif self.onnx:
            LOGGER.info(f'Loading {weights} for ONNX Runtime inference...')
            import onnxruntime

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(weights, providers=providers)
            self.output_names = [x.name for x in self.session.get_outputs()]

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16

        if self.pt:  # PyTorch
            with torch.no_grad():
                y = self.model(im)
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        if self.device.type != 'cpu':
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        typpes = ["pt", "onnx"]
        typpes = [s in Path(p).name for s in typpes]
        return typpes
