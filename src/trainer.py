import os
import random
import sys

sys.path.append("./src")
import numpy as np
import torch
from dataloader.augmentations import AugmentAlbumentations, ResizeAlbumentations
from dataloader.dataloader import ListDataset
from models.loss import ComputeLoss
from models.yolo import Model
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.optim import smart_optimizer
from utils.pretrain import intersect_dicts
from utils.utils import check_yaml
from validater import compute_metric


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
    train_transform = AugmentAlbumentations(size=args.img_size)
    train_dataset = ListDataset(
        list_path=args.trainset,
        img_size=args.img_size,
        transform=train_transform,
        use_cache=args.cache,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        worker_init_fn=worker_seed_set,
    )
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
    return train_dataloader, val_dataloader


def train(args):
    # ====================================== initialize basic information for training ========
    writer = SummaryWriter(args.tensorboard)
    if not os.path.exists(args.parameter_dir):
        os.makedirs(args.parameter_dir)
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_mp, best_mr, best_map50, best_map = 0, 0, 0, 0
    start_epoch = 1

    # ============================== load classes, configuration, and create dataloader ========
    classes = load_classes(args.cocos)
    train_dataloader, val_dataloader = create_dataloder(args)
    cfg = check_yaml(args.cfg)  # check YAML

    # ====================================== create model ======================================
    model = Model(cfg=cfg).to(device)

    # ====================================== create loss function ===============================
    hyper_parameters = {
        "box": args.box,  # box loss gain,
        "cls": args.cls,  # cls loss gain,
        "cls_pw": args.cls_pw,  # cls BCELoss positive_weight
        "obj": args.obj,  # obj loss gain (scale with pixels)
        "obj_pw": args.obj_pw,  # obj BCELoss positive_weight
        "iou_t": args.iou_t,  # IoU training threshold
        "anchor_t": args.anchor_t,  # anchor-multiple threshold
        "fl_gamma": args.fl_gamma,  # focal loss gamma"
    }
    compute_loss = ComputeLoss(
        model, hyper_parameters=hyper_parameters
    )  # init loss class

    # ====================================== set optimizer/scheduler ============================
    optimizer = smart_optimizer(
        model,
        name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.EPOCH // 3, gamma=0.1
    )

    # ====================================== mixed precision training ===========================
    scaler = torch.cuda.amp.GradScaler()

    # ====================================== initialized model ==================================
    # load checkpoint to CPU to avoid CUDA memory leak
    if os.path.exists(args.pretrained):
        print(args.pretrained)
        ckpt = torch.load(args.pretrained, map_location='cpu')

        # checkpoint state_dict as FP32
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=[])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        print("initialized model parameters by {}".format(args.pretrained))
    if os.path.exists(args.continuous):
        status = torch.load(args.continuous)
        start_epoch = status["epoch"]
        model.load_state_dict(status["model_state_dict"])
        optimizer.load_state_dict(status["optimizer_state_dict"])
        scheduler.load_state_dict(status["scheduler_state_dict"])
        print("continuous training by {}".format(args.continuous))
    torch.save(model.state_dict(), args.parameter_dir + "/yolo.ckpt")
    # ====================================== train ==================================
    for epoch in range(start_epoch, args.EPOCH + 1):
        with tqdm(train_dataloader, desc=f"Training Epoch {epoch}") as _tqdm:
            model.train()
            for batch_i, (imgs, targets) in enumerate(train_dataloader):
                batches_done = len(train_dataloader) * epoch + batch_i
                imgs = imgs.type(torch.FloatTensor).to(device)
                targets = targets.to(device)
                if args.amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pred = model(imgs)
                        loss, loss_items = compute_loss(pred, targets.to(device))
                        loss = loss / args.accumulate_steps
                        scaler.scale(loss).backward()
                        if batches_done % args.accumulate_steps == 0:
                            scaler.unscale_(optimizer)  # unscale gradients
                            clip_grad_norm_(
                                model.parameters(), max_norm=10.0
                            )  # clip gradients
                            scaler.step(optimizer)  # optimizer.step
                            scaler.update()
                            optimizer.zero_grad()
                else:
                    pred = model(imgs)
                    loss, loss_items = compute_loss(pred, targets.to(device))
                    loss = loss / args.accumulate_steps
                    loss.backward()
                    if batches_done % args.accumulate_steps == 0:
                        clip_grad_norm_(
                            model.parameters(), max_norm=10.0
                        )  # clip gradients
                        optimizer.step()
                        optimizer.zero_grad()

                _tqdm.set_postfix_str(
                    "loss {:.4f}, iou_loss {:.4f}, obj_loss {:.4f}, class_loss {:.4f}, lr {:.10f}".format(
                        loss.item(),
                        loss_items[0],
                        loss_items[1],
                        loss_items[2],
                        scheduler.get_last_lr()[0],
                    )
                )
                _tqdm.update(1)
                writer.add_scalar('loss/total_loss', loss, batches_done)
                writer.add_scalar('loss/iou_loss', loss_items[0], batches_done)
                writer.add_scalar('loss/obj_loss', loss_items[1], batches_done)
                writer.add_scalar('loss/class_loss', loss_items[2], batches_done)

        # ====================================== validate ==================================
        if epoch % args.val_interval == 0:
            torch.save(
                model.state_dict(),
                args.checkpoints + "/yolo_{}.ckpt".format(epoch),
            )
            mp, mr, map50, map = compute_metric(
                model,
                val_dataloader,
                args.conf_thres,
                args.iou_thres,
                args.max_det,
                classes,
                args.amp,
            )
            if best_map < map:
                torch.save(model.state_dict(), args.parameter_dir + "/yolo.ckpt")
                best_mp, best_mr, best_map50, best_map = mp, mr, map50, map
            print(
                "best_mp {:.4f}, best_mr {:.4f}, best_map50 {:.4f}, best_map {:.4f}".format(
                    best_mp, best_mr, best_map50, best_map
                )
            )

        # ====================================== save intermedia status =============================
        # save interval state for continues training
        if args.continuous:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                },
                args.continuous,
            )
        scheduler.step()
