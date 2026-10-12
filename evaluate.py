import torch

from models.unet3d import UNet3D
from utils.dataloader import get_dataloader
from utils.metrics import dice_score, iou_score
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet3D().to(device)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.eval()

loader = get_dataloader()

images, labels = next(iter(loader))

images = images.float().to(device)

with torch.no_grad():

    prediction = model(images)

prediction = torch.sigmoid(prediction)
prediction = (prediction > 0.5).float()

dice = dice_score(prediction.cpu(), labels)
iou = iou_score(prediction.cpu(), labels)

print("=" * 40)
print("MODEL EVALUATION")
print("=" * 40)
print(f"Dice Score : {dice:.4f}")
print(f"IoU Score  : {iou:.4f}")
print("=" * 40)
import json
import os
evaluation_results = {
    "dice_score": float(dice),
    "iou_score": float(iou)
}

os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/evaluation.json",
    "w"
) as f:

    json.dump(
        evaluation_results,
        f,
        indent=4
    )

print("\nEvaluation saved to:")
print("outputs/evaluation.json")