from monai.transforms import (
    Compose,
    ScaleIntensity,
    EnsureType
)


def get_image_transforms():

    return Compose([
        ScaleIntensity(),
        EnsureType()
    ])


def get_label_transforms():

    return Compose([
        EnsureType()
    ])