import torch
from tqdm import tqdm
from utils.metrics import ap_per_class, non_max_suppression, process_batch, xywh2xyxy


def evaluation(model, dataloader, conf_thres, iou_thres, max_det, classes):
    model.eval()
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
            nb, _, height, width = imgs.shape  # batch size, channels, height, width
            targets = targets.to(device)
            with torch.no_grad():
                preds, train_out = model(imgs)
            # NMS
            # to pixels
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
            lb = []  # for autolabelling
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False, max_det=max_det
            )
            # Metrics
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                # number of labels, predictions
                nl, npr = labels.shape[0], pred.shape[0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                # seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
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
        print(map)

    return None
