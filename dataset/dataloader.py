import os
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def resize(image: Any, size: int):
    """Resize the images

    Args:
        image (numpy.array): the images will be resized
        size (int): image's size

    Returns:
        numpy.array: return the resized images
    """
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def img2label_paths(img_paths: List[str]) -> List[str]:
    """Convert the path of images into path of labels

    Args:
        img_paths (List[str]): list of each image's path

    Returns:
        List[str]: return the list of path of each image's label
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


class ListDataset(Dataset):
    r"""Dataset for coco data

    Return:
        img : normalized images with (channel, img_size, img_size)
        boxes : [
                [one-hot, box_x, box_y , box_w, box_h],
                [one-hot, box_x, box_y , box_w, box_h],
                [one-hot, box_x, box_y , box_w, box_h],
                ]
    """

    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.img_files = [item.strip() for item in self.img_files]
        self.label_files = img2label_paths(self.img_files)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):
        img = cv2.imread(self.img_files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = np.loadtxt(self.label_files[index]).reshape(-1, 5)
        boxes = self.__check__(boxes)
        img, boxes = self.transform((img, boxes))
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = img / 255.0

        return img, boxes

    def __len__(self):
        return len(self.img_files)

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        imgs = torch.stack([torch.from_numpy(img) for img in imgs])
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

    def __check__(self, boxes):
        """check the boxes is valid or not

        Args:
            boxes (numpy.array): normalized boxes

        Returns:
            numpy.array: return valid boxes if the boxes are out of (1.0, 0.0)
        """
        up_bound = 1.0
        low_bound = 0.0
        boxes[:, 1:] = np.clip(boxes[:, 1:], low_bound, up_bound)
        return boxes
