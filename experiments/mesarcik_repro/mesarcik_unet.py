"""U-Net used as the "U-Net" baseline in Mesarcik et al. (2022), copied verbatim.

Source: https://github.com/mesarcik/RFI-NLN, models.py (commit 9e756de),
functions Conv2D_block and UNET. MIT License, Copyright (c) 2021 Michael
(Misha) Mesarcik -- see LICENSE_RFI-NLN in this directory.

Only change: `args.input_shape` is passed as a plain tuple argument instead of
an argparse Namespace. The layer graph is untouched.

NOTE: this is NOT Akeret et al.'s tf_unet. It is the authors' own Keras U-Net:
same padding, stride-2 convolutions instead of max pooling, BatchNorm,
dropout 0.05, 16 base filters, sigmoid output.
"""
import tensorflow as tf
from tensorflow.keras import layers


def Conv2D_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, stride=(1, 1)):
    # first layer
    x = layers.Conv2D(filters=n_filters,
                      kernel_size=(kernel_size, kernel_size),
                      kernel_initializer='he_normal',
                      strides=stride,
                      padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def UNET(input_shape, n_filters=16, dropout=0.05, batchnorm=True):
    # Contracting Path
    input_data = tf.keras.Input(input_shape, name='data')
    if input_shape[0] == 16: _str = 1  # cant downsample 16x16 patches
    else: _str = 2
    c1 = Conv2D_block(input_data, n_filters * 1, kernel_size=3, batchnorm=batchnorm,
                      stride=(_str, _str))
    p1 = layers.Dropout(dropout)(c1)

    c2 = Conv2D_block(p1, n_filters * 2, kernel_size=3, stride=(2, 2), batchnorm=batchnorm)
    p2 = layers.Dropout(dropout)(c2)

    if input_shape[1] > 8:
        c3 = Conv2D_block(p2, n_filters * 4, kernel_size=3, stride=(2, 2), batchnorm=batchnorm)
        p3 = layers.Dropout(dropout)(c3)
    else: p3 = p2

    if input_shape[1] > 16:
        c4 = Conv2D_block(p3, n_filters * 8, kernel_size=3, stride=(2, 2), batchnorm=batchnorm)
        p4 = layers.Dropout(dropout)(c4)
    else: p4 = p3

    c5 = Conv2D_block(p4, n_filters=n_filters * 16, kernel_size=3, stride=(2, 2),
                      batchnorm=batchnorm)

    # Expansive Path
    if input_shape[1] > 16:
        u6 = layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        u6 = layers.Dropout(dropout)(u6)
        c6 = Conv2D_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    else: c6 = c5

    if input_shape[1] > 8:
        u7 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        u7 = layers.Dropout(dropout)(u7)
        c7 = Conv2D_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    else: c7 = c6

    u8 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(dropout)(u8)
    c8 = Conv2D_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(dropout)(u9)
    if input_shape[0] != 16:  # cant downsample 16x16 patches
        u9 = layers.UpSampling2D((2, 2))(u9)
    c9 = Conv2D_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model
