import pickle
import time

import numpy as np
from keras import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import *
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def custom_generator(X, y, batch_size):
    aug = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

    gen_reg = aug.flow(X, y['regression_output'], batch_size=batch_size, seed=1)
    gen_cls = aug.flow(X, y['classification_output'], batch_size=batch_size, seed=1)

    while True:
        X = gen_reg.next()
        Y = gen_cls.next()

        yield X[0], {'regression_output': X[1], 'classification_output': Y[1]}


train_data = np.load('training_data_reg_cls_shanghai.npy', allow_pickle=True)

X = np.array([i[0] for i in train_data])
X = X / 255.0

y = np.array([i[1] for i in train_data])  # regression
z = np.array([i[2] for i in train_data])  # classification

NAME = "custom_cnn_reg_cls_{}".format(int(time.time()))

(trainX, testX, trainY, testY, trainZ, testZ) = train_test_split(X, y, z, test_size=0.25, random_state=42)

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
# model = Dropout(0.25)(model)

# 8
model = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(model)
model = BatchNormalization()(model)
model = LeakyReLU()(model)

model = Conv2D(filters=15, kernel_size=(1, 1), activation="linear", padding="same")(model)

# 8
model = GlobalAveragePooling2D()(model)

##############################################################
regression_output = Dense(1, activation="relu", name='regression_output')(model)

classification_output = Dense(15, activation='softmax', name='classification_output')(model)

# construct the CNN
model = Model(inputs, [regression_output, classification_output])

loss = {
    'regression_output': 'mean_squared_error',
    'classification_output': 'sparse_categorical_crossentropy'
}

loss_weights = {
    'regression_output': 10,
    'classification_output': 0.1
}

metrics = {
    'regression_output': 'mean_absolute_error',
    'classification_output': 'accuracy'
}

model.compile(optimizer='adam', loss=loss, loss_weights=loss_weights, metrics=metrics)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

monitor = 'val_classification_output_accuracy'
checkpoint = ModelCheckpoint("{}_best.weights".format(NAME), monitor=monitor, verbose=1,
                             save_best_only=True, save_weights_only=True)

# model.summary()
plot_model(model, to_file="{}_model.pdf".format(NAME), show_shapes=True)

model_json = model.to_json()
with open("{}_model.json".format(NAME), 'w') as json_file:
    json_file.write(model_json)

# train the model
history = model.fit_generator(
    custom_generator(trainX, {'regression_output': trainY, 'classification_output': trainZ}, batch_size=32),
    validation_data=(testX, {'regression_output': testY, 'classification_output': testZ}),
    steps_per_epoch=len(trainX) // 32, epochs=1000, callbacks=[tensorboard, checkpoint], verbose=0
)

model.save("{}.model".format(NAME))

with open("{}.hist".format(NAME), 'wb') as hist:
    pickle.dump(history.history, hist)
