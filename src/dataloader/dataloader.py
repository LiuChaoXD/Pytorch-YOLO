import os
import random
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.logger import LOGGER
from utils.utils import colorstr


class ListDataset(Dataset):
    """ListDataset for coco dataset

    Parameters
    ----------
        list_path (list[string]) : list of each image's path
        img_size (int) : image size
        multiscale (bool) : whether to use mutliscale training strategy
        transform (torchvision.transforms.Compose) : the transform list
        use_cache (bool) : whether use cache to boost the data load

    Methods
    ----------
        collate_fn(batch) :

        examples:
        >>>  dataset = ListDataset(...)
        >>>  dataloader = DataLoader(..., collate_fn=dataset.collate_fn, ...)
    """

    def __init__(self, list_path, img_size=640, transform=None, use_cache=False):
        # process the images' and labels' text information
        with open(list_path, "r") as file:
            img_files = file.readlines()
        img_files = [item.strip() for item in img_files]
        label_files = self.__img2label(img_files)
        self.img_files, self.label_files = self.__check_exist(img_files, label_files)

        # determine the type of cache
        self.img_size = img_size
        self.use_cache = use_cache
        if use_cache:
            self.cache_type = 'ram' if self.__check_cache_ram() else 'npy'
            if self.cache_type == 'npy':
                self.npy_files = [Path(f).with_suffix('.npy') for f in self.img_files]

        # set the transform
        self.transform = transform

        # cache images
        if self.use_cache:
            if self.cache_type == 'ram':
                self.cache_imgs = []
                for index in tqdm(range(len(self.img_files)), desc="Caching data"):
                    img = cv2.imread(self.img_files[index])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.cache_imgs.append(img)
                if len(self.cache_imgs) != len(self.img_files):
                    raise ValueError("Some images are not cached! Please check...")

            elif self.cache_type == 'npy':
                for index in tqdm(range(len(self.img_files)), desc="Caching data"):
                    f = self.npy_files[index]
                    if not f.exists():
                        img = cv2.imread(self.img_files[index])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        np.save(f.as_posix(), img)
            else:
                raise ValueError("Cache errors!")

            # cache labels
            self.cache_boxes = [
                np.loadtxt(self.label_files[index]).reshape(-1, 5) for index in range(len(self.label_files))
            ]

    def __getitem__(self, index):
        if not self.use_cache:
            img = cv2.imread(self.img_files[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = np.loadtxt(self.label_files[index]).reshape(-1, 5)
            boxes = self.__check_box(boxes)
        else:
            if self.cache_type == 'ram':
                img = self.cache_imgs[index]
            elif self.cache_type == 'npy':
                img = np.load(self.npy_files[index])
            boxes = self.cache_boxes[index]

        img, boxes = self.transform(img, boxes)
        img = torch.from_numpy(img.transpose((2, 0, 1)) / 255.0)
        return img, boxes

    def __len__(self):
        return len(self.img_files)

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        imgs = torch.stack(imgs)
        # Add sample index to targets
        bs_boxes = []
        for i, boxes in enumerate(targets):
            if boxes is None:
                continue
            boxes = np.insert(boxes, 0, values=i, axis=1)
            bs_boxes.append(boxes)
        # Remove empty placeholder targets
        bs_boxes = [torch.from_numpy(boxes) for boxes in bs_boxes if boxes is not None]
        bs_boxes = torch.cat(bs_boxes, 0)
        return imgs, bs_boxes

    def __check_box(self, boxes, epison=1e-10):
        """Check the boxes is valid or not

        Args:
            boxes (numpy.array): normalized boxes

        Returns:
            numpy.array: return valid boxes if the boxes are out of (0.0, 1.0]
        """
        up_bound = 1.0
        low_bound = 0.0 + epison
        boxes[:, 1:] = np.clip(boxes[:, 1:], low_bound, up_bound)
        return boxes

    def __check_exist(self, img_paths, label_paths, verbose=True):
        """Checking the image's path and its corresponding label's path are existed

        Args:
            img_paths (list[string]): list of each path to image
            label_paths (list[string]): list of each label path to image

        Returns:
            (valid_img_paths, valid_label_paths):
        """
        valid_img_paths, valid_label_paths = [], []
        for img, label in zip(img_paths, label_paths):
            if os.path.exists(img) and os.path.exists(label):
                valid_img_paths.append(img)
                valid_label_paths.append(label)
            else:
                if verbose:
                    dir = str(Path(img).parent)
                    name = Path(img).stem
                    LOGGER.info(
                        f"{colorstr('Data:')} {dir}/{name}{Path(img).suffix} or {Path(label).suffix} is not existed. "
                        f"{colorstr('Drop it!')}"
                    )
                continue
        LOGGER.info(
            f"{colorstr('Background images:')} there are {(len(img_paths) - len(valid_img_paths))/len(img_paths)*100:.2f}% background images are removed. "
        )
        return valid_img_paths, valid_label_paths

    def __img2label(self, img_paths: List[str]) -> List[str]:
        """Convert the path of images into path of labels

        Args:
            img_paths (List[str]): list of each image's path

        Returns:
            List[str]: return the list of path of each image's label
        """
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

    def __check_cache_ram(self, safety_margin=0.1):
        """Check cache size is larger than avaiable cache

        Args:
            safety_margin (float, optional): _description_. Defaults to 0.1.

        Returns:
            bool: True or False
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(len(self.img_files), 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.img_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * len(self.img_files) / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available
        LOGGER.info(f"{colorstr('Cache:')} require {mem_required/gb:.2f} GB, available size {mem.available/gb:.2f}. ")
        return cache
