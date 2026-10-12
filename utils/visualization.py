import matplotlib.pyplot as plt
import os


class Visualizer:

    def __init__(self, output_folder="outputs"):
        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

    def save_ct_slice(self, image, slice_number, filename):

        plt.figure(figsize=(6,6))
        plt.imshow(image[:, :, slice_number], cmap="gray")
        plt.title("CT Scan")
        plt.axis("off")

        save_path = os.path.join(self.output_folder, filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def save_mask(self, mask, slice_number, filename):

        plt.figure(figsize=(6,6))
        plt.imshow(mask[:, :, slice_number], cmap="jet")
        plt.title("Segmentation Mask")
        plt.axis("off")

        save_path = os.path.join(self.output_folder, filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def save_overlay(self, image, mask, slice_number, filename):

        plt.figure(figsize=(6,6))

        plt.imshow(image[:, :, slice_number], cmap="gray")
        plt.imshow(mask[:, :, slice_number], cmap="jet", alpha=0.4)

        plt.title("Overlay")
        plt.axis("off")

        save_path = os.path.join(self.output_folder, filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()