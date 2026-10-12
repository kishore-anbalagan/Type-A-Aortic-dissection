import os
import nibabel as nib
from torch.utils.data import Dataset

from utils.preprocessing import preprocess_image, preprocess_label
from utils.transforms import get_image_transforms, get_label_transforms


class ImageTAADDataset(Dataset):

    def __init__(self, dataset_path, transform=None):

        self.dataset_path = dataset_path

        self.images_path = os.path.join(
            dataset_path,
            "imagesTr"
        )

        self.labels_path = os.path.join(
            dataset_path,
            "labelsTr"
        )

        # Get image files
        image_files = [
            f for f in os.listdir(self.images_path)
            if f.endswith(".nii.gz")
        ]

        # Get label files
        label_files = [
            f for f in os.listdir(self.labels_path)
            if f.endswith(".nii.gz")
        ]

        # Create dictionary:
        # TAAD_101 -> TAAD_101.nii.gz
        label_dict = {}

        for label in label_files:

            patient_id = label.replace(".nii.gz", "")

            label_dict[patient_id] = label

        # Match images with corresponding labels
        self.image_files = []
        self.label_files = []

        for image in image_files:

            # TAAD_101_0000.nii.gz
            patient_id = image.replace(
                "_0000.nii.gz",
                ""
            )

            if patient_id in label_dict:

                self.image_files.append(image)

                self.label_files.append(
                    label_dict[patient_id]
                )

        # Sort matched pairs together
        pairs = sorted(
            zip(
                self.image_files,
                self.label_files
            )
        )

        self.image_files = [
            pair[0] for pair in pairs
        ]

        self.label_files = [
            pair[1] for pair in pairs
        ]

        print("\nMatched Image-Label Pairs:")

        for i in range(min(10, len(self.image_files))):

            print(
                self.image_files[i],
                " <--> ",
                self.label_files[i]
            )

        print(
            "\nTotal Matched Pairs:",
            len(self.image_files)
        )

        # Transforms
        self.image_transform = get_image_transforms()

        self.label_transform = get_label_transforms()

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        image_path = os.path.join(
            self.images_path,
            self.image_files[idx]
        )

        label_path = os.path.join(
            self.labels_path,
            self.label_files[idx]
        )

        image = nib.load(
            image_path
        ).get_fdata()

        label = nib.load(
            label_path
        ).get_fdata()

        # Preprocessing
        image = preprocess_image(image)

        label = preprocess_label(label)

        # Transforms
        image = self.image_transform(image)

        label = self.label_transform(label)

        return image, label