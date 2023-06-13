import random
import sys

import numpy as np
import torch
from dataset.augmentations import DefaultAug, NoAug
from dataset.dataloader import ListDataset
from models.loss import ComputeLoss
from models.yolo import Model
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from utils.optim import smart_optimizer
from utils.pretrain import intersect_dicts
from utils.utils import check_yaml
from validater import evaluation


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


def create_dataloder(args):
    dataset = ListDataset(
        args.img_path,
        img_size=args.img_size,
        multiscale=False,
        transform=transforms.Compose(
            [
                NoAug(args.img_size, normalized=True),
            ]
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set,
    )

    val_dataset = ListDataset(
        args.val_img_path,
        img_size=args.img_size,
        multiscale=False,
        transform=transforms.Compose(
            [
                NoAug(args.img_size, normalized=True),
            ]
        ),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        worker_init_fn=worker_seed_set,
    )
    return dataloader, val_dataloader


def train(args):
    train_dataloader, val_dataloader = create_dataloder(args)
    classes = load_classes(args.cocos)

    cfg = check_yaml(args.cfg)  # check YAML
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(cfg=cfg).to(device)

    compute_loss = ComputeLoss(model)  # init loss class
    optimizer = smart_optimizer(model, name='Adam', lr=0.0001, momentum=0.937, decay=0.0005)

    # load checkpoint to CPU to avoid CUDA memory leak
    ckpt = torch.load(args.pretrained, map_location='cpu')
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=[])  # intersect
    model.load_state_dict(csd, strict=False)  # load

    for epoch in range(1, args.EPOCH + 1):
        with tqdm(train_dataloader, desc=f"Training Epoch {epoch}") as _tqdm:
            model.train()
            for batch_i, (imgs, targets) in enumerate(train_dataloader):
                batches_done = len(train_dataloader) * epoch + batch_i
                imgs = imgs.type(torch.FloatTensor).to(device)
                targets = targets.to(device)
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))
                loss.backward()
                # if batches_done % yolo.hyperparams['accumulated_step'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                _tqdm.set_postfix_str(
                    "loss {:.4f}, iou_loss {:.4f}, obj_loss {:.4f}, class_loss {:.4f}, lr {}".format(
                        loss.item(),
                        loss_items[0],
                        loss_items[1],
                        loss_items[2],
                        optimizer.param_groups[0]["lr"],
                    )
                )
                _tqdm.update(1)
        if epoch % args.val_interval == 0:
            evaluation(model, val_dataloader, args.conf_thres, args.iou_thres, args.max_det, classes)
