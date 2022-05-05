"""
Custom data generator to work with BraTS2020 dataset.
Can be used as a template
loads data from local directory
in batches. 
"""
import os
import numpy as np


def load_image(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):
        if image_name.split(".")[1] == "npy":

            image = np.load(img_dir + image_name)
            images.append(image)

    images = np.array(images)

    return images


def image_loader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    # keras need the generator infinite, so we will use while true
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:

            limit = min(batch_end, L)

            X = load_image(img_dir, img_list[batch_start:limit])
            Y = load_image(mask_dir, mask_list[batch_start:limit])

            yield (X, Y)  # A tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


###################################################################

# Test the generators

from matplotlib import pyplot as plt
import random

train_img_dir = "BraTS2020_TrainingData/input_data_3channels/images/"
train_msk_dir = "BraTS2020_TrainingData/input_data_3channels/masks/"
tran_img_list = os.listdir(train_img_dir)
train_msk_list = os.listdir(train_msk_dir)

batch_size = 2

train_img_datagen = image_loader(
    train_img_dir, tran_img_list, train_msk_dir, train_msk_list, batch_size
)

# Verify generator
img, msk = train_img_datagen.__next__()

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_msk = msk[img_num]
test_msk = np.argmax(test_msk, axis=3)

n_slice = random.randint(0, test_msk.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap="gray")
plt.title("Image flair")
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap="gray")
plt.title("Image t1ce")
plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap="gray")
plt.title("Image t2")
plt.subplot(224)
plt.imshow(test_msk[:, :, n_slice])
plt.title("Mask")
plt.show()
