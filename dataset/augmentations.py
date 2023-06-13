import random
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def xywh2xyxy(input):
    """Convert boxing's xywh coordinates into xyxy

    Args:
        input (numpy.array): input is [x, y, w, h]

    Returns:
        numpy.array: boxing's xyxy coordinates
    """
    output = np.copy(input)
    output[..., 0] = input[..., 0] - input[..., 2] / 2
    output[..., 1] = input[..., 1] - input[..., 3] / 2
    output[..., 2] = input[..., 0] + input[..., 2] / 2
    output[..., 3] = input[..., 1] + input[..., 3] / 2
    return output


class RelativeBox(object):
    """Convert absolute coordinates into relative coordinates (normalized coordinates)

    Parameters
    ----------

    Examples
    --------
    >>> relativeBox = RelativeBox()
    >>> data = (img, boxes)
    >>> img, boxes = relativeBox(data)
    """

    def __init__(self):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes


class AbsoluteBox(object):
    """Convert the relative normalized

    [x_min, y_min, x_center, y_center]

    to

    [x_min * image_width, y_min * image_height, x_center * image_width, y_center * image_height]

    Parameters
    ----------

    Examples
    --------
    >>> absoluteBox = AbsoluteBox()
    >>> data = (img, boxes)
    >>> img, boxes = absoluteBox(data)
    """

    def __init__(self):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes


class ImgAug(object):
    """Images augmentations

    Parameters
    ----------
    augmentations (iaa.Sequential): list of the augmentations (iaa.Sequential)
    normalized (bool) : normalized the box and coordinates

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> augmentations = iaa.Sequential(...)
    >>> normalized = True
    >>> imgAug = ImgAug(augmentations, normalized)
    >>> data = (img, boxes)
    >>> img, boxes = imgAug(data)
    """

    def __init__(self, augmentations=[], normalized=False):
        self.augmentations = augmentations
        self.normalized = normalized

    def __absolute_box(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return boxes

    def __normalized_box(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return boxes

    def __call__(self, data):
        # Unpack data
        img, boxes = data
        # convert normalized boxes into absolute box
        boxes = self.__absolute_box(data)

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy(boxes[:, 1:])

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes], shape=img.shape
        )

        # Apply augmentations
        img, bounding_boxes = self.augmentations(image=img, bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = (x1 + x2) / 2
            boxes[box_idx, 2] = (y1 + y2) / 2
            boxes[box_idx, 3] = x2 - x1
            boxes[box_idx, 4] = y2 - y1
        if self.normalized:
            boxes = self.__normalized_box((img, boxes))
        return img, boxes


class DefaultAug(ImgAug):
    """Default images augmentations

    Parameters
    ----------
    img_size (int): size of final images.
    nnormalized (bool) : normalized the box and coordinates.
    """

    def __init__(self, img_size, normalized=True):
        self.augmentations = iaa.Sequential(
            [
                iaa.Sharpen((0.0, 0.1)),
                iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
                iaa.AddToBrightness((-60, 40)),
                iaa.AddToHue((-10, 10)),
                iaa.Fliplr(0.5),
                iaa.Resize(size=(img_size, img_size)),
            ]
        )
        self.normalized = normalized


class NoAug(ImgAug):
    """Default images augmentations. Just contains resize operation.

    Parameters
    ----------
    img_size (int): size of final images.
    normalized (bool) : normalized the box and coordinates.
    """

    def __init__(self, img_size, normalized):
        self.augmentations = iaa.Sequential(
            [
                iaa.Resize(size=(img_size, img_size)),
            ]
        )
        self.normalized = normalized
