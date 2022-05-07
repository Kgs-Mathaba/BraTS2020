# Converts the standard 2d unet to 3d
from ast import Lambda
from keras.models import Model
from keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    concatenate,
    Conv3DTranspose,
    BatchNormalization,
    Dropout,
    Lambda,
)
from keras.optimizers import adam_v2 as Adam
from keras.metrics import MeanIoU

kernel_initializer = "he_uniform"  # Can try others

########################################################


def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))

    # Contraction path
    c1 = Conv3D(
        16,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(
        16,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(
        32,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(
        32,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(
        128,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(
        128,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    c5 = Conv3D(
        256,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(
        256,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(c5)

    # Expansive path
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(
        128,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(
        128,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(
        32,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(
        32,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(
        16,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(
        16,
        (3, 3, 3),
        activation="relu",
        kernel_initializer=kernel_initializer,
        padding="same",
    )(c9)

    outputs = Conv3D(num_classes, (1, 1, 1), activation="softmax")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # Compile model outside this function to make it flexible
    model.summary()

    return model


if __name__ == "__main__":
    # Test model construction
    model = simple_unet_model(128, 128, 128, 3, 4)
    print(model.input_shape)
    print(model.output_shape)
