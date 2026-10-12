import torch
from monai.losses import DiceLoss
from torch.optim import Adam
from tqdm import tqdm
import os

from models.unet3d import UNet3D
from utils.dataloader import get_dataloader
import config


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# ============================================================
# MODEL
# ============================================================

model = UNet3D().to(device)


# ============================================================
# DATA LOADER
# ============================================================

train_loader = get_dataloader()


# ============================================================
# LOSS FUNCTION
# ============================================================

loss_function = DiceLoss(sigmoid=True)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)


# ============================================================
# BEST LOSS
# ============================================================

best_loss = float("inf")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ============================================================
# TRAINING
# ============================================================

print("\nTraining Started...\n")


for epoch in range(config.EPOCHS):

    model.train()

    epoch_loss = 0.0

    progress = tqdm(
        train_loader,
        total=len(train_loader)
    )

    for images, labels in progress:

        # Move data to device
        images = images.float().to(device)
        labels = labels.float().to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = loss_function(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Add loss
        epoch_loss += loss.item()

        # Update progress bar
        progress.set_description(
            f"Epoch {epoch + 1}/{config.EPOCHS}"
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    # ========================================================
    # AVERAGE LOSS
    # ========================================================

    avg_loss = (
        epoch_loss /
        len(train_loader)
    )

    print(
        f"\nEpoch {epoch + 1} Average Loss : "
        f"{avg_loss:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "outputs/unet_model.pth"
        )

        print(
            "Best Model Saved!"
        )


# ============================================================
# TRAINING COMPLETED
# ============================================================

print(
    "\nTraining Completed Successfully!"
)

print(
    "\nBest Model Path:"
)

print(
    "outputs/unet_model.pth"
)