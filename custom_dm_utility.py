import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Lambda, Concatenate, Conv2D, Add, multiply, Dense, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Reshape
from tensorflow.keras.utils import Sequence

from db_mean_std import *

np.random.seed(123)
random.seed(123)
tf.random.set_seed(1234)


class DataGenerator(Sequence):

    def __init__(self, dataframe, bench_name, batch_size, im_size, dm_factor, mode, scale, is_dm, is_lm):
        self.batch_size = batch_size
        self.dm_factor = dm_factor
        self.out_size = im_size // dm_factor
        self.mode = mode
        self.scale = scale
        self.is_dm = is_dm
        self.is_lm = is_lm

        if not self.is_dm and not self.is_lm:
            raise Exception("both density map and point map cannot be disabled")

        # depends on dataset
        self.mean, self.std = get_mean_std(INPUT_SIZE=im_size, bench_name=bench_name)

        if self.mean is None or self.std is None:
            raise Exception("pixel mean and std values are not given")

        self.x_col = dataframe['file_name'].tolist()
        if is_dm:
            self.y_col = dataframe['dm_name'].tolist()
        if is_lm:
            self.z_col = dataframe['lm_name'].tolist()

        self.indexes = np.arange(len(self.x_col))

    def __len__(self):
        # compute number of batches to yield
        return len(self.x_col) // self.batch_size

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        if self.mode == 'train':
            np.random.shuffle(self.indexes)

    def get_batch(self, idx_start, idx_end):
        im_list = self.x_col[idx_start: idx_end]
        dm_list = []
        if self.is_dm:
            dm_list = self.y_col[idx_start: idx_end]
        lm_list = []
        if self.is_lm:
            lm_list = self.z_col[idx_start: idx_end]

        batch_im = []
        batch_dm = []
        batch_lm = []
        for it, im in enumerate(im_list):
            im = cv2.imread(im)
            im = im / 255
            for c in range(3):
                im[:, :, c] = (im[:, :, c] - self.mean[c]) / self.std[c]

            if self.scale > 1:
                im = cv2.resize(im, dsize=(0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

            mirror = False
            if random.uniform(0, 1) >= 0.5:
                im = np.flip(im, axis=1)
                mirror = True
            batch_im.append(im)

            if self.is_dm:
                dm = np.load(dm_list[it], allow_pickle=True)
                if self.dm_factor > 1:
                    dm = cv2.resize(dm, (self.out_size, self.out_size),
                                    interpolation=cv2.INTER_AREA) * self.dm_factor * self.dm_factor
                if mirror:
                    dm = np.flip(dm, axis=1)
                batch_dm.append(dm)

            if self.is_lm:
                lm = np.load(lm_list[it], allow_pickle=True)
                if mirror:
                    lm = np.flip(lm, axis=1)
                batch_lm.append(lm)

        batch_im = np.array(batch_im)
        if self.is_dm:
            batch_dm = np.array(batch_dm)
            batch_dm = np.expand_dims(batch_dm, axis=-1)
        if self.is_lm:
            batch_lm = np.array(batch_lm)
            batch_lm = np.expand_dims(batch_lm, axis=-1)

        return batch_im, batch_dm, batch_lm

    def __getitem__(self, idx):
        idx_start = idx * self.batch_size
        idx_end = (idx + 1) * self.batch_size

        batch_x, batch_y, batch_z = self.get_batch(idx_start, idx_end)
        if self.is_dm and self.is_lm:
            return batch_x, {'density_map_output': batch_y, 'point_map_output': batch_z}
        elif self.is_dm:
            return batch_x, batch_y
        elif self.is_lm:
            return batch_x, batch_z
        else:
            raise Exception("both density map and point map cannot be disabled")


def init_from_vgg(model, freeze):
    vgg = VGG16()

    vgg_weights = []
    for layer in vgg.layers:
        name = layer.name
        if 'conv' in name:
            vgg_weights.append(layer.get_weights())

    offset = 0
    i = 0
    while i < 10:
        idx = i + offset
        name = model.layers[idx].name
        if 'conv' in name:
            model.layers[idx].set_weights(vgg_weights[i])
            if freeze:
                model.layers[idx].trainable = False
            i = i + 1
        else:
            offset = offset + 1

    return model


def counting_error(y_true, y_pred):
    return K.sum(y_true) - K.sum(y_pred)


def pixel_error(y_true, y_pred):
    return y_true - y_pred + 1e-8


# el
def euclidean_distance_loss(y_true, y_pred):
    # get Euclidean distance loss
    return K.sqrt(K.sum(K.square(pixel_error(y_true, y_pred)), axis=-1))


# msec
def mean_squared_error_count(y_true, y_pred):
    # get squared count from density map
    return K.mean(K.square(counting_error(y_true, y_pred)))


# msep
def mean_squared_error_image(y_true, y_pred):
    # get squared error from density map
    return K.mean(K.square(pixel_error(y_true, y_pred)), axis=-1)


# maec
def mean_absolute_error_count(y_true, y_pred):
    # get absolute count from density map
    return K.abs(counting_error(y_true, y_pred))


# msep
def mean_absolute_error_image(y_true, y_pred):
    # get absolute error from density map
    return K.mean(K.abs(pixel_error(y_true, y_pred)), axis=-1)


# bcep
def binary_crossentropy_image(y_true, y_pred):
    return K.sum(-100 * y_true * K.log(y_pred + 1e-8) - (1 - y_true) * K.log(1 - y_pred + 1e-8), axis=-1)


def log10(x):
    numerator = K.log(x)
    denominator = K.log(K.constant(10, dtype=numerator.dtype))
    return numerator / denominator


# psnr_acc
def psnr_accuracy(y_true, y_pred):
    # get PSNR from density map with axis
    mse = K.mean(K.square(pixel_error(y_true, y_pred)), axis=-1)
    psnr = 20 * log10(1 / K.sqrt(mse))
    return psnr


def binarize_images(y_true, y_pred):
    kernel = [[0.0, 0.2, 0.0],
              [0.2, 0.2, 0.2],
              [0.0, 0.2, 0.0]]
    kernel = K.constant(kernel, dtype='float32', shape=[3, 3, 1, 1])
    y_true = K.conv2d(x=y_true, kernel=kernel, padding="same")
    y_true = K.round(y_true)

    y_pred = K.pool2d(x=y_pred, pool_size=(3, 3), strides=(1, 1), padding="same", pool_mode="avg")
    y_max = K.pool2d(x=y_pred, pool_size=(3, 3), strides=(1, 1), padding="same", pool_mode="max")
    y_max = K.equal(y_max, y_pred)
    y_max = K.cast(y_max, dtype='float32')
    y_pred *= y_max
    y_pred = K.round(y_pred)

    return y_true, y_pred


# lce
def localized_counting_error(y_true, y_pred):
    y_true, y_pred = binarize_images(y_true, y_pred)
    return K.abs(counting_error(y_true, y_pred))


# lpe
def localized_percentage_error(y_true, y_pred):
    y_true, y_pred = binarize_images(y_true, y_pred)
    y_true, y_pred = K.sum(y_true), K.sum(y_pred)
    error = K.abs(y_true - y_pred)
    return error / y_true * 100


def steep_sigmoid(x):
    return 1 / (1 + K.exp(1 - 10 * x))


def mish(x):
    return x * K.tanh(K.softplus(x))


def conv_attention_module(feature, init):
    feature = channel_attention(feature, init)
    feature = spatial_attention(feature, init)
    return feature


def channel_attention(input_feature, init, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer=init)
    shared_layer_two = Dense(channel, kernel_initializer=init)

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature, init):
    kernel_size = 7

    cbam_feature = input_feature
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer=init,
                          use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])


def loss_index(short_code):
    if short_code == 'maep':
        loss = mean_absolute_error_image
    elif short_code == 'msep':
        loss = mean_squared_error_image
    elif short_code == 'bcep':
        loss = binary_crossentropy_image
    elif short_code == 'el':
        loss = euclidean_distance_loss
    else:
        raise Exception("undefined loss function")

    return loss
