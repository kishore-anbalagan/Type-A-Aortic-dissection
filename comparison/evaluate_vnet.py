import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference

from models.vnet import VNet
from utils.dataloader import ImageTAADDataset


# ==============================
# SETTINGS
# ==============================

DATASET_PATH = (
    r"C:\PROJECT AI\IMAGETAAD DATASET"
    r"\imageTAAD\ImageTAAD"
    r"\Task068_TAAD"
)

MODEL_PATH = r"outputs\vnet\vnet_model.pth"

NUM_IMAGES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================
# DICE
# ==============================

def dice_score(pred, target):

    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()

    return (
        2.0 * intersection /
        (pred.sum() + target.sum() + 1e-8)
    ).item()


# ==============================
# IOU
# ==============================

def iou_score(pred, target):

    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()

    union = (
        pred.sum()
        + target.sum()
        - intersection
    )

    return (
        intersection /
        (union + 1e-8)
    ).item()


# ==============================
# MAIN
# ==============================

print("\n===================================")
print("          V-NET EVALUATION")
print("===================================")

print("Using device:", DEVICE)


# Check model
if not os.path.exists(MODEL_PATH):

    print("\nERROR: V-Net model not found!")
    print(MODEL_PATH)
    sys.exit()


# Load dataset
print("\nLoading dataset...")

dataset = ImageTAADDataset(DATASET_PATH)

print(
    "\nEvaluating only",
    NUM_IMAGES,
    "images for quick testing."
)


# Load model
model = VNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print("\nV-Net model loaded successfully.")


# ==============================
# EVALUATION
# ==============================

dice_scores = []
iou_scores = []


with torch.no_grad():

    for index in tqdm(
        range(min(NUM_IMAGES, len(dataset))),
        desc="Evaluating"
    ):

        images, labels = dataset[index]

        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)

        images = images.float().to(DEVICE)
        labels = labels.float().to(DEVICE)

        outputs = sliding_window_inference(
            inputs=images,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25
        )

        outputs = torch.sigmoid(outputs)

        prediction = (
            outputs > 0.5
        ).float()

        dice = dice_score(
            prediction,
            labels
        )

        iou = iou_score(
            prediction,
            labels
        )

        dice_scores.append(dice)
        iou_scores.append(iou)

        print(
            f"\nImage {index + 1}: "
            f"Dice = {dice:.4f}, "
            f"IoU = {iou:.4f}"
        )


# ==============================
# FINAL RESULT
# ==============================

mean_dice = np.mean(dice_scores)
mean_iou = np.mean(iou_scores)


print("\n===================================")
print("       V-NET EVALUATION RESULT")
print("===================================")

print(
    f"Mean Dice Score : {mean_dice:.4f}"
)

print(
    f"Mean IoU Score  : {mean_iou:.4f}"
)

print(
    f"Evaluated Images: {len(dice_scores)}"
)

print("===================================")

print("\nEvaluation completed successfully.")