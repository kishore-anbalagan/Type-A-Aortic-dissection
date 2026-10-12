import torch
import torch.nn.functional as F


def preprocess_image(image):

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    image = image.unsqueeze(0).unsqueeze(0)

    image = F.interpolate(
        image,
        size=(128, 128, 128),
        mode="trilinear",
        align_corners=False
    )

    return image.squeeze(0)


def preprocess_label(label):

    label = torch.tensor(
        label,
        dtype=torch.float32
    )

    # Convert to binary mask
    label = (label > 0).float()

    label = label.unsqueeze(0).unsqueeze(0)

    image = F.interpolate(
        label,
        size=(128, 128, 128),
        mode="nearest"
    )

    return image.squeeze(0)