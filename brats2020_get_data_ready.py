import numpy as np
import nibabel as nib
import glob
from pandas import test
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("WARNING")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

##########################
# This part of the code to get an initial understanding of the dataset.
#################################
# PART 1: Load sample images and visualize
# Includes, dividing each image by its max to scale them to [0,1]
# Converting mask from float to uint8
# Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
# Visualize
###########################################
# View a few images

# Note: Segmented file name in Folder 355 has a weird name. Rename it to match others.

TRAIN_DATASET_PATH = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
VALIDATION_DATASET_PATH = "BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/"

test_image_flair = nib.load(
    TRAIN_DATASET_PATH + "BraTS20_Training_355/BraTS20_Training_355_flair.nii"
).get_fdata()
# Scalers are applied to 1D so let us reshape and then reshape back to original shape.
test_image_flair = scaler.fit_transform(
    test_image_flair.reshape(-1, test_image_flair.shape[-1])
).reshape(test_image_flair.shape)

test_image_t1 = nib.load(
    TRAIN_DATASET_PATH + "BraTS20_Training_355/BraTS20_Training_355_t1.nii"
).get_fdata()
test_image_t1 = scaler.fit_transform(
    test_image_t1.reshape(-1, test_image_t1.shape[-1])
).reshape(test_image_t1.shape)

test_image_t1ce = nib.load(
    TRAIN_DATASET_PATH + "BraTS20_Training_355/BraTS20_Training_355_t1ce.nii"
).get_fdata()
test_image_t1ce = scaler.fit_transform(
    test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])
).reshape(test_image_t1.shape)

test_image_t2 = nib.load(
    TRAIN_DATASET_PATH + "BraTS20_Training_355/BraTS20_Training_355_t2.nii"
).get_fdata()
test_image_t2 = scaler.fit_transform(
    test_image_t2.reshape(-1, test_image_t1.shape[-1])
).reshape(test_image_t2.shape)

test_mask = nib.load(
    TRAIN_DATASET_PATH + "BraTS20_Training_355/BraTS20_Training_355_seg.nii"
).get_fdata()
test_mask = test_mask.astype(np.uint8)

print(np.unique(test_mask))  # 0, 1, 2, 4 (Need to reencode to 0, 1, 2, 3)
print(test_mask.shape)
test_mask[test_mask == 4] = 3

import random

n_slice = random.randint(0, test_mask.shape[2])
# print("n_slice = ", n_slice)
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_image_flair[:, :, n_slice], cmap="gray")
plt.title("Image Flair")

plt.subplot(232)
plt.imshow(test_image_t1[:, :, n_slice], cmap="gray")
plt.title("Image t1")

plt.subplot(233)
plt.imshow(test_image_t1ce[:, :, n_slice], cmap="gray")
plt.title("Image t1ce")

plt.subplot(234)
plt.imshow(test_image_t2[:, :, n_slice], cmap="gray")
plt.title("Image t2")

plt.subplot(235)
plt.imshow(test_mask[:, :, n_slice], cmap="gray")
plt.title("Mask")

plt.show()


##################################################
# PART 2: Explore the process of combining images to channels and divide them to patches
# Includes...
# Combining all 4 images to 4 channels of a numpy array.
#
################################################
# Flair, T1CE, annd T2 have the most information
# Combine t1ce, t2, and flair into single multichannel image

combined_x = np.stack([test_image_flair, test_image_t1ce, test_image_t2], axis=3)
print("combined x shape = ", combined_x.shape)

# Crop to size to be divisible by 64 so we can later extract 64x64x64 patches
# cropping x,y and z
# combined_x = combined_x[24:216, 24:216, 13:141]

combined_x = combined_x[56:184, 56:184, 13:141]  # Crop to 128x128x128x4

# Do the same for mask
test_mask = test_mask[56:184, 56:184, 13:141]
n_slice = random.randint(0, test_mask.shape[2])

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_image_flair[:, :, n_slice], cmap="gray")
plt.title("Image Flair")

plt.subplot(232)
plt.imshow(test_image_t1[:, :, n_slice], cmap="gray")
plt.title("Image t1")

plt.subplot(233)
plt.imshow(test_image_t1ce[:, :, n_slice], cmap="gray")
plt.title("Image t1ce")

plt.subplot(234)
plt.imshow(test_image_t2[:, :, n_slice], cmap="gray")
plt.title("Image t2")

plt.subplot(235)
plt.imshow(test_mask[:, :, n_slice], cmap="gray")
plt.title("Mask")

plt.show()

imsave("BraTS2020_TrainingData/combined255.tif", combined_x)
np.save("BraTS2020_TrainingData/combined255.npy", combined_x)
# Verify image is being read properly
my_image = np.load("BraTS2020_TrainingData/combined255.npy")
print("my image shape = ", my_image.shape)

test_mask = to_categorical(test_mask, num_classes=4)
####################################################################
#####################################
# End of understanding the dataset. Now get it organized.
#####################################

# Now let us apply the same as above to all the images...
# Merge channels, crop, patchify, save
# GET DATA READY =  GENERATORS OR OTHERWISE

# Keras datagenerator does not support 3d
### images lists
# t1_list = sorted(glob.glob(TRAIN_DATASET_PATH + "*/*t1.nii"))
t2_list = sorted(glob.glob(TRAIN_DATASET_PATH + "*/*t2.nii"))
t1ce_list = sorted(glob.glob(TRAIN_DATASET_PATH + "*/*t1ce.nii"))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH + "*/*flair.nii"))
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH + "*/*seg.nii"))

# Each volume generates 18 64x64x64x4 sub-volumes.
# Total 369 volumes = 6642 sub volumes

for img in range(len(t2_list)):
    print("Now preparing image and mask number: ", img)

    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    temp_image_t2 = scaler.fit_transform(
        temp_image_t2.reshape(-1, temp_image_t2.shape[-1])
    ).reshape(temp_image_t2.shape)

    temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(
        temp_image_t2.reshape(-1, temp_image_t1ce.shape[-1])
    ).reshape(temp_image_t1ce.shape)

    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(
        temp_image_flair.reshape(-1, temp_image_flair.shape[-1])
    ).reshape(temp_image_flair.shape)

    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask == 4] = 3  # reaasign mask values 4 to 3

    temp_combined_images = np.stack(
        [temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3
    )

    # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches
    # Cropping x,y,z
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    # print("Temp image shape = ", temp_combined_images.shape)
    val, counts = np.unique(temp_mask, return_counts=True)

    if (
        1 - (counts[0] / counts.sum())
    ) > 0.01:  # At least 1% useful volume with labels that are not 0
        print("Save Me")
        temp_mask = to_categorical(temp_mask, num_classes=4)
        np.save(
            "BraTS2020_TrainingData/input_data_3channels/images/image_"
            + str(img)
            + ".npy",
            temp_combined_images,
        )
        np.save(
            "BraTS2020_TrainingData/input_data_3channels/masks/mask_"
            + str(img)
            + ".npy",
            temp_mask,
        )
    else:
        print("I am useless")

##########################################################################
# Reapet process for validation set
### validation images lists
# val_t1_list = sorted(glob.glob(VALIDATION_DATASET_PATH + "*/*t1.nii"))
val_t2_list = sorted(glob.glob(VALIDATION_DATASET_PATH + "*/*t2.nii"))
val_t1ce_list = sorted(glob.glob(VALIDATION_DATASET_PATH + "*/*t1ce.nii"))
val_flair_list = sorted(glob.glob(VALIDATION_DATASET_PATH + "*/*flair.nii"))
# val_mask_list = sorted(glob.glob(VALIDATION_DATASET_PATH + "*/*seg.nii"))


for val_img in range(len(val_t2_list)):
    print("Now preparing validation image: ", val_img)

    val_temp_image_t2 = nib.load(val_t2_list[val_img]).get_fdata()
    val_temp_image_t2 = scaler.fit_transform(
        val_temp_image_t2.reshape(-1, val_temp_image_t2.shape[-1])
    ).reshape(val_temp_image_t2.shape)

    val_temp_image_t1ce = nib.load(val_t1ce_list[val_img]).get_fdata()
    val_temp_image_t1ce = scaler.fit_transform(
        val_temp_image_t1ce.reshape(-1, val_temp_image_t1ce.shape[-1])
    ).reshape(val_temp_image_t1ce.shape)

    val_temp_image_flair = nib.load(val_flair_list[val_img]).get_fdata()
    val_temp_image_flair = scaler.fit_transform(
        val_temp_image_flair.reshape(-1, val_temp_image_flair.shape[-1])
    ).reshape(val_temp_image_flair.shape)

    val_temp_combined_images = np.stack(
        [val_temp_image_flair, val_temp_image_t1ce, val_temp_image_t2], axis=3
    )

    # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches
    # Cropping x,y,z
    val_temp_combined_images = val_temp_combined_images[56:184, 56:184, 13:141]
    # print("Val image shape = ", val_temp_combined_images.shape)
    np.save(
        "BraTS2020_ValidationData/input_data_3channels/images/image_"
        + str(val_img)
        + ".npy",
        val_temp_combined_images,
    )
