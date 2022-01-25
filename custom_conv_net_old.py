import pickle
import time

import numpy as np
from keras import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import *
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# change
reg = False

train_data = np.load('training_data_reg_cls.npy', allow_pickle=True)

X = np.array([i[0] for i in train_data])
X = X / 255.0

if reg:
    y = np.array([i[1] for i in train_data])
    NAME = "custom_cnn_reg_{}".format(int(time.time()))
else:
    y = np.array([i[2] for i in train_data])
    NAME = "custom_cnn_cls_{}".format(int(time.time()))

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.25, random_state=42)

aug = ImageDataGenerator(rotation_range=40,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

inputs = Input(shape=X.shape[1:])

##############################################################
# 128
model = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(inputs)
model = BatchNormalization()(model)
model = LeakyReLU()(model)
model = MaxPooling2D(pool_size=(2, 2))(model)

# 64
model = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)
model = MaxPooling2D(pool_size=(2, 2))(model)

# 32
model = SeparableConv2D(filters=128, kernel_size=(3, 3), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)

model = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)

model = SeparableConv2D(filters=128, kernel_size=(3, 3), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)
model = MaxPooling2D(pool_size=(2, 2))(model)
if not reg:
    model = Dropout(0.25)(model)

# 16
model = SeparableConv2D(filters=256, kernel_size=(3, 3), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)

model = Conv2D(filters=128, kernel_size=(1, 1), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)

model = SeparableConv2D(filters=256, kernel_size=(3, 3), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)

model = Conv2D(filters=128, kernel_size=(1, 1), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)

model = SeparableConv2D(filters=256, kernel_size=(3, 3), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)
model = MaxPooling2D(pool_size=(2, 2))(model)
if not reg:
    model = Dropout(0.5)(model)

# 8
model = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)

model = Conv2D(filters=15, kernel_size=(1, 1), activation="linear", padding="same")(model)

# 8
model = GlobalAveragePooling2D()(model)

##############################################################
if reg:
    model = Dense(1, activation="relu")(model)
else:
    model = Dense(15, activation="softmax")(model)

# construct the CNN
model = Model(inputs, model)

if reg:
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    monitor = 'val_mean_absolute_error'
else:
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    monitor = 'val_accuracy'

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

checkpoint = ModelCheckpoint("{}_best.weights".format(NAME), monitor=monitor, verbose=1,
                             save_best_only=True, save_weights_only=True)

# model.summary()
plot_model(model, to_file="{}_model.pdf".format(NAME), show_shapes=True)

model_json = model.to_json()
with open("{}_model.json".format(NAME), 'w') as json_file:
    json_file.write(model_json)

# train the model
history = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                              validation_data=(testX, testY),
                              steps_per_epoch=len(trainX) // 32, epochs=500,
                              callbacks=[tensorboard, checkpoint])

model.save("{}.model".format(NAME))

with open("{}.hist".format(NAME), 'wb') as hist:
    pickle.dump(history.history, hist)
