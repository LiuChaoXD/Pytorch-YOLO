import random
from typing import Any

import cv2
import numpy as np
import psutil
from utils.logger import LOGGER
from utils.utils import colorstr


class Cache:
    def __init__(self, img_files, img_size, cache_dir) -> None:
        self.cache_dir = cache_dir
        self.img_files, self.img_size = img_files, img_size
        self.cache_flag = self.__check_cache_ram()

    def __check_cache_ram(self, safety_margin=0.1):
        # Check image caching requirements vs available memory
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(len(self.img_files), 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.img_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = (
            b * len(self.img_files) / n
        )  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available
        LOGGER.info(
            f"{colorstr('green', 'Cache:')} require {mem_required/(1<<30):.2f} GB, available size {mem.available/(1<<30):.2f}. "
            f"{colorstr('green', 'Cannot Cache')}"
        )
        return cache

    def cache_data(self) -> Any:
        return None
