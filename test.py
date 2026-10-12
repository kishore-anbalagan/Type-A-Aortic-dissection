import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.unet3d import UNet3D
from utils.dataloader import get_dataloader
import config

# Create output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

print("Loading model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet3D().to(device)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.eval()

print("Model Loaded!")

loader = get_dataloader()

print("Loading one image...")

images, labels = next(iter(loader))

images = images.float().to(device)
labels = labels.float().to(device)

print("Running Prediction...")

with torch.no_grad():
    prediction = model(images)

print("Prediction Completed!")

# Convert prediction to binary mask
prediction = torch.sigmoid(prediction)
prediction = (prediction > 0.5).float()

# ====================================================
# Save complete 3D prediction
# ====================================================
prediction_np = prediction[0, 0].cpu().numpy()
np.save("outputs/prediction.npy", prediction_np)

print("3D Prediction saved as outputs/prediction.npy")

# ====================================================
# Save visualization (middle slice)
# ====================================================
slice_idx = prediction_np.shape[2] // 2

image = images[0, 0, :, :, slice_idx].cpu().numpy()
label = labels[0, 0, :, :, slice_idx].cpu().numpy()
pred = prediction_np[:, :, slice_idx]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("CT Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(label, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred, cmap="gray")
plt.title("Prediction")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/prediction.png", dpi=300)
plt.close()

print("Prediction image saved in outputs/prediction.png")
print("Testing Completed Successfully!")