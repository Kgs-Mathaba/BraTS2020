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
print("n_slice = ", n_slice)
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
