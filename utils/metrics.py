import torch


def dice_score(pred, target):

    pred = pred.float()
    target = target.float()

    smooth = 1e-5

    intersection = (pred * target).sum()

    dice = (2. * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

    return dice.item()


def iou_score(pred, target):

    pred = pred.float()
    target = target.float()

    smooth = 1e-5

    intersection = (pred * target).sum()

    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()