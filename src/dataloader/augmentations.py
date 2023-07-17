import albumentations as A
import numpy as np


class AugmentAlbumentations:
    def __init__(self, size=640):
        augment_transformer = [
            A.Blur(p=0.2),
            A.MedianBlur(p=0.2),
            A.Flip(0.2),
            A.ToGray(p=0.2),
            A.CLAHE(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.ImageCompression(quality_lower=75, p=0.2),
            A.Cutout(num_holes=84, max_h_size=8, max_w_size=8, fill_value=0, p=0.2),
            A.MultiplicativeNoise(
                multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2
            ),
            A.ToSepia(p=0.2),
            A.RandomSizedBBoxSafeCrop(height=size, width=size, erosion_rate=0.0, p=1.0),
        ]
        self.transform = A.Compose(
            augment_transformer,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
        )

    def __call__(self, im, labels):
        new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])
        im, labels = new['image'], np.array(
            [[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]
        )
        return im, labels


class ResizeAlbumentations:
    def __init__(self, size=640):
        self.transform = A.Compose(
            [A.Resize(height=size, width=size)],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
        )

    def __call__(self, im, labels):
        new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])
        im, labels = new['image'], np.array(
            [[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]
        )
        return im, labels
