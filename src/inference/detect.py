import torch.nn as nn
import torch
from pathlib import Path
from utils.utils import colorstr, LOGGER
import onnxruntime


class DetectMultiBackend(nn.Module):
    def __init__(self, weights='./yolo.onnx', fp16=False, engine='ONNX') -> None:
        super().__init__()
        self.fp16 = fp16
        cuda = True if torch.cuda.is_available() else False
        if engine == "ONNX":
            assert Path(weights).suffix == ".onnx"
            LOGGER.info(f'Loading {weights} for ONNX Runtime inference...')
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(weights, providers=providers)
            self.output_names = [x.name for x in self.session.get_outputs()]

    def forward(self, im):
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        im = im.cpu().numpy()  # torch to numpy
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
