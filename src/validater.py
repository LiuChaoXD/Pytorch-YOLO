import os
import random
import sys

sys.path.append("./src")
import numpy as np
import torch
from dataloader.augmentations import ResizeAlbumentations
from dataloader.dataloader import ListDataset
from models.yolo import Model
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import LOGGER
from utils.metrics import ap_per_class, non_max_suppression, process_batch, xywh2xyxy
from utils.pretrain import intersect_dicts
from utils.utils import check_yaml, colorstr


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        names = fp.read().splitlines()
    names = dict(enumerate(names))
    return names


def worker_seed_set(worker_id):
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def compute_metric(model, dataloader, conf_thres, iou_thres, max_det, classes, amp):
    model.eval()
    model.half() if amp else model.float()

    # im = im.half() if half else im.float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = []
    stats, ap, ap_class = [], [], []
    # iou vector for mAP@0.5:0.95
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    with tqdm(dataloader, desc=f"Evaluating") as _tqdm:
        for batch_i, (imgs, targets) in enumerate(dataloader):
            # obtain the metric value of training
            imgs = imgs.type(torch.FloatTensor).to(device)
            (nb, _, height, width) = imgs.shape  # batch size, channels, height, width
            targets = targets.to(device)
            imgs = imgs.half() if amp else imgs.float()  # uint8 to fp16/32
            with torch.no_grad():
                preds, train_out = model(imgs)
            # NMS
            # to pixels
            targets[:, 2:] *= torch.tensor(
                (width, height, width, height), device=device
            )
            lb = []  # for autolabelling
            preds = non_max_suppression(
                preds,
                conf_thres,
                iou_thres,
                labels=lb,
                multi_label=True,
                agnostic=False,
                max_det=max_det,
            )
            # Metrics
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                # number of labels, predictions
                nl, npr = labels.shape[0], pred.shape[0]
                correct = torch.zeros(
                    npr, niou, dtype=torch.bool, device=device
                )  # init
                # seen += 1

                if npr == 0:
                    if nl:
                        stats.append(
                            (correct, *torch.zeros((2, 0), device=device), labels[:, 0])
                        )
                    continue

                # Predictions
                predn = pred.clone()

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    correct = process_batch(predn, labelsn, iouv)
                # (correct, conf, pcls, tcls)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
            _tqdm.update(1)
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=classes)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    return mp, mr, map50, map


def evaluation(args):
    val_transform = ResizeAlbumentations(size=args.img_size)
    val_dataset = ListDataset(
        list_path=args.valset,
        img_size=args.img_size,
        transform=val_transform,
        use_cache=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 4,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        worker_init_fn=worker_seed_set,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = check_yaml(args.cfg)  # check YAML
    classes = load_classes(args.cocos)

    # ====================================== create model ======================================
    model = Model(cfg=cfg).to(device)
    # load checkpoint to CPU to avoid CUDA memory leak
    if os.path.exists(args.inference):
        # model.load_state_dict(torch.load(args.inference))
        ckpt = torch.load(args.inference, map_location='cpu')
        # checkpoint state_dict as FP32
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=[])  # intersect
        model.load_state_dict(csd, strict=False)  # load
    mp, mr, map50, map = compute_metric(
        model,
        val_dataloader,
        args.conf_thres,
        args.iou_thres,
        args.max_det,
        classes,
        args.amp,
    )
    LOGGER.info(
        f"{colorstr('MP:')} {mp * 100:.4f}, {colorstr('MR:')} {mr * 100:.4f}, {colorstr('MAP50:')} {map50 * 100:.4f}, {colorstr('MAP:')} {map * 100:.4f}"
    )
    return None
