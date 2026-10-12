import os
import json
import numpy as np
import nibabel as nib

# ==============================
# PATHS
# ==============================

PREDICTION_PATH = "outputs/prediction.npy"

# Change this to your actual CT image
# Example:
#CT_IMAGE_PATH = r"C:\PROJECT AI\TypeA_Aortic_Dissection\imagesTr\TAAD_101_0000.nii.gz"
CT_IMAGE_PATH = r"C:\PROJECT AI\IMAGETAAD DATASET\imageTAAD\ImageTAAD\Task068_TAAD\imagesTr\TAAD_101_0000.nii.gz"
OUTPUT_PATH = "outputs/measurements.json"


# ==============================
# CHECK FILES
# ==============================

if not os.path.exists(PREDICTION_PATH):
    print("Error: prediction.npy not found!")
    print("Run test.py first.")
    exit()

if not os.path.exists(CT_IMAGE_PATH):
    print("Error: CT image not found!")
    print("Check CT_IMAGE_PATH in measurements.py")
    exit()


# ==============================
# LOAD PREDICTION
# ==============================

mask = np.load(PREDICTION_PATH)

# Remove unnecessary dimensions
mask = np.squeeze(mask)

print("Prediction Shape:", mask.shape)


# ==============================
# LOAD ORIGINAL CT
# ==============================

ct_image = nib.load(CT_IMAGE_PATH)

# Get actual voxel spacing
spacing = ct_image.header.get_zooms()[:3]

print("Voxel Spacing:", spacing)


# ==============================
# CREATE BINARY MASK
# ==============================

mask = (mask > 0).astype(np.uint8)

coords = np.argwhere(mask)

if len(coords) == 0:
    print("No segmented region detected.")
    exit()


# ==============================
# VOXEL COUNT
# ==============================

voxel_count = int(np.sum(mask))


# ==============================
# VOLUME
# ==============================

voxel_volume_mm3 = (
    spacing[0] *
    spacing[1] *
    spacing[2]
)

volume_mm3 = voxel_count * voxel_volume_mm3
volume_cm3 = volume_mm3 / 1000


# ==============================
# BOUNDING BOX
# ==============================

min_coords = coords.min(axis=0)
max_coords = coords.max(axis=0)

length_voxels = max_coords[0] - min_coords[0] + 1
width_voxels = max_coords[1] - min_coords[1] + 1
depth_voxels = max_coords[2] - min_coords[2] + 1


# Convert voxel dimensions to mm

length_mm = length_voxels * spacing[0]
width_mm = width_voxels * spacing[1]
depth_mm = depth_voxels * spacing[2]

maximum_diameter_mm = max(
    length_mm,
    width_mm,
    depth_mm
)


# ==============================
# CENTROID
# ==============================

centroid_voxel = coords.mean(axis=0)

centroid_mm = [
    float(centroid_voxel[0] * spacing[0]),
    float(centroid_voxel[1] * spacing[1]),
    float(centroid_voxel[2] * spacing[2])
]


# ==============================
# PRINT RESULTS
# ==============================

print("\n===================================")
print("DISEASE MEASUREMENTS")
print("===================================")

print(f"Segmented Voxels : {voxel_count}")

print(f"Volume (mm³)     : {volume_mm3:.2f}")
print(f"Volume (cm³)     : {volume_cm3:.2f}")

print(f"Length (mm)      : {length_mm:.2f}")
print(f"Width (mm)       : {width_mm:.2f}")
print(f"Depth (mm)       : {depth_mm:.2f}")

print(f"Maximum Diameter : {maximum_diameter_mm:.2f} mm")

print(f"Centroid (mm)    : {centroid_mm}")

print("===================================")


# ==============================
# SAVE RESULTS
# ==============================

results = {

    "segmented_voxels": voxel_count,

    "voxel_spacing_mm": [
        float(spacing[0]),
        float(spacing[1]),
        float(spacing[2])
    ],

    "volume_mm3": float(volume_mm3),

    "volume_cm3": float(volume_cm3),

    "length_mm": float(length_mm),

    "width_mm": float(width_mm),

    "depth_mm": float(depth_mm),

    "maximum_diameter_mm": float(maximum_diameter_mm),

    "centroid_mm": centroid_mm
}


os.makedirs("outputs", exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=4)


print("\nMeasurements saved to:")
print(OUTPUT_PATH)