import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vnet import VNet
from utils.dataloader import ImageTAADDataset


DATASET_PATH = (
    r"C:\PROJECT AI\IMAGETAAD DATASET"
    r"\imageTAAD\ImageTAAD"
    r"\Task068_TAAD"
)

OUTPUT_DIR = "outputs/vnet"

EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 0.0001


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

dataset = ImageTAADDataset(
    DATASET_PATH
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = VNet().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("\n===================================")
print("          V-NET TRAINING")
print("===================================")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=loss.item()
        )

    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"\nEpoch {epoch + 1} "
        f"Average Loss : {average_loss:.4f}"
    )


model_path = os.path.join(
    OUTPUT_DIR,
    "vnet_model.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

print("\nV-Net training completed.")

print(
    "Model saved to:",
    model_path
)