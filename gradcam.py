import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.unet3d import UNet3D
from utils.dataloader import get_dataloader


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "outputs/unet_model.pth"
OUTPUT_DIR = "outputs/gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model not found!")
    print("Expected:", MODEL_PATH)
    exit()

model = UNet3D().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()

print("3D U-Net loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading CT image...")

loader = get_dataloader()

images, labels = next(iter(loader))

images = images.float().to(device)
labels = labels.float().to(device)

print("Input shape:", images.shape)


# ============================================================
# GRAD-CAM STORAGE
# ============================================================

activation = None
gradient = None


def forward_hook(module, input, output):
    global activation
    activation = output


def backward_hook(module, grad_input, grad_output):
    global gradient
    gradient = grad_output[0]


# ============================================================
# FIND LAST CONV3D LAYER
# ============================================================

target_layer = None

for module in model.modules():

    if isinstance(module, torch.nn.Conv3d):
        target_layer = module


if target_layer is None:

    print("ERROR: Conv3D layer not found.")
    exit()


print("Grad-CAM target layer found.")


forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# FORWARD PASS
# ============================================================

model.zero_grad()

output = model(images)

print("Model output shape:", output.shape)


# ============================================================
# TARGET SEGMENTATION OUTPUT
# ============================================================

prediction_probability = torch.sigmoid(output)

target = prediction_probability.mean()

target.backward()

print("Gradients calculated.")


# ============================================================
# REMOVE HOOKS
# ============================================================

forward_handle.remove()
backward_handle.remove()


# ============================================================
# CHECK RESULTS
# ============================================================

if activation is None or gradient is None:

    print("ERROR: Grad-CAM failed.")
    exit()


print("Activation shape:", activation.shape)
print("Gradient shape:", gradient.shape)


# ============================================================
# CALCULATE GRAD-CAM
# ============================================================

weights = gradient.mean(
    dim=(2, 3, 4),
    keepdim=True
)

cam = (
    weights * activation
).sum(dim=1, keepdim=True)

cam = torch.relu(cam)


# ============================================================
# UPSAMPLE GRAD-CAM TO INPUT SIZE
# ============================================================

cam = torch.nn.functional.interpolate(
    cam,
    size=images.shape[2:],
    mode="trilinear",
    align_corners=False
)


# ============================================================
# NORMALIZE GRAD-CAM
# ============================================================

cam_min = cam.min()
cam_max = cam.max()

cam = (
    cam - cam_min
) / (
    cam_max - cam_min + 1e-8
)


# ============================================================
# NUMPY
# ============================================================

cam = cam.detach().cpu().numpy()[0, 0]

image = images.detach().cpu().numpy()[0, 0]

prediction = prediction_probability.detach().cpu().numpy()[0, 0]


# ============================================================
# CREATE BINARY SEGMENTATION
# ============================================================

segmentation = prediction > 0.5


# ============================================================
# SELECT SLICE
# ============================================================

slice_index = image.shape[2] // 2

image_slice = image[:, :, slice_index]

cam_slice = cam[:, :, slice_index]

prediction_slice = prediction[:, :, slice_index]

segmentation_slice = segmentation[:, :, slice_index]


# ============================================================
# IMPROVE HEATMAP VISIBILITY
# ============================================================

# Only show stronger activations
threshold = 0.60

important_region = cam_slice.copy()

important_region[important_region < threshold] = 0


# ============================================================
# SAVE GRAD-CAM ARRAY
# ============================================================

cam_path = os.path.join(
    OUTPUT_DIR,
    "gradcam.npy"
)

np.save(
    cam_path,
    cam
)

print("\nGrad-CAM saved:", cam_path)


# ============================================================
# 1. GRAD-CAM HEATMAP
# ============================================================

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_heatmap.png"
)

plt.figure(figsize=(8, 8))

plt.imshow(
    image_slice,
    cmap="gray"
)

plt.imshow(
    important_region,
    cmap="jet",
    alpha=0.65
)

plt.axis("off")

plt.title(
    "3D U-Net Grad-CAM Important Regions"
)

plt.savefig(
    heatmap_path,
    bbox_inches="tight",
    dpi=200
)

plt.close()


# ============================================================
# 2. CT + GRAD-CAM OVERLAY
# ============================================================

overlay_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_overlay.png"
)

plt.figure(figsize=(8, 8))

plt.imshow(
    image_slice,
    cmap="gray"
)

plt.imshow(
    cam_slice,
    cmap="jet",
    alpha=0.45
)

plt.contour(
    segmentation_slice,
    levels=[0.5],
    linewidths=2
)

plt.axis("off")

plt.title(
    "CT + Grad-CAM + Segmentation Boundary"
)

plt.savefig(
    overlay_path,
    bbox_inches="tight",
    dpi=200
)

plt.close()


# ============================================================
# 3. SEGMENTATION + GRAD-CAM
# ============================================================

seg_overlay_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_segmentation_overlay.png"
)

plt.figure(figsize=(8, 8))

plt.imshow(
    image_slice,
    cmap="gray"
)

# Segmentation boundary
plt.contour(
    segmentation_slice,
    levels=[0.5],
    linewidths=2
)

# Strong Grad-CAM region
plt.imshow(
    important_region,
    cmap="jet",
    alpha=0.60
)

plt.axis("off")

plt.title(
    "Segmentation + Important Grad-CAM Regions"
)

plt.savefig(
    seg_overlay_path,
    bbox_inches="tight",
    dpi=200
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n===================================")
print("       GRAD-CAM COMPLETED")
print("===================================")

print("\nGenerated files:")

print(
    "1. Grad-CAM Heatmap:"
)
print(
    heatmap_path
)

print(
    "\n2. CT + Grad-CAM Overlay:"
)
print(
    overlay_path
)

print(
    "\n3. Segmentation + Grad-CAM:"
)
print(
    seg_overlay_path
)

print(
    "\n4. Grad-CAM Array:"
)
print(
    cam_path
)