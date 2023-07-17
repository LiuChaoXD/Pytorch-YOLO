import glob
import math
import os
import sys
import urllib
from copy import deepcopy
from pathlib import Path

import pkg_resources as pkg
import thop
import torch
import torch.nn as nn
from torchinfo import summary

from .logger import LOGGER

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv3 Detect() module m, and correct if necessary
    # mean anchor area per output layer
    a = m.anchors.prod(-1).mean(-1).view(-1)
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def model_info(model, verbose=False, imgsz=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )  # number gradients
    if verbose:
        print(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print(
                '%5g %40s %9s %12g %20s %10.3g %10.3g'
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                )
            )

    try:  # FLOPs
        p = next(model.parameters())
        stride = (
            max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        )  # max stride
        # input image in BCHW format
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
        flops = (
            thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1e9 * 2
        )  # stride GFLOPs
        imgsz = (
            imgsz if isinstance(imgsz, list) else [imgsz, imgsz]
        )  # expand if int/float
        # 640x640 GFLOPs
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'
    except Exception:
        fs = ''

    name = (
        Path(model.yaml_file).stem.replace('yolov5', 'YOLOv3')
        if hasattr(model, 'yaml_file')
        else 'Model'
    )
    LOGGER.info(
        f"{colorstr(name)} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}"
    )


def check_version(
    current='0.0.0',
    minimum='0.0.0',
    name='version ',
    pinned=False,
    hard=False,
    verbose=False,
):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    # string
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv3, but {name}{current} is currently installed'
    if hard:
        assert result  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    # color arguments, string
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m',
    }
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}'


def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=''):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(
            urllib.parse.unquote(file).split('?')[0]
        ).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            # file already exists
            LOGGER.info(f'Found {url} locally at {file}')
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert (
                Path(file).exists() and Path(file).stat().st_size > 0
            ), f'File download failed: {url}'  # check
        return file
    elif file.startswith('clearml://'):  # ClearML Dataset ID
        assert (
            'clearml' in sys.modules
        ), "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:  # search
        files = []
        for d in 'data', 'models', 'utils':  # search directories
            files.extend(
                glob.glob(str(ROOT / d / '**' / file), recursive=True)
            )  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        # assert unique
        assert (
            len(files) == 1
        ), f"Multiple files match '{file}', specify exact path: {files}"
        return files[0]  # return file


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor
