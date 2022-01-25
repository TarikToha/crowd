import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.layers import BatchNormalization, LeakyReLU, Conv2D, MaxPooling2D, UpSampling2D, \
    AveragePooling2D
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils.vis_utils import plot_model

from custom_dm_utility import *

resume = False
num_epochs = 100
bench_name = 'ucf_18_hajj'
dataset_path = '/content/drive/My Drive/traffic_signal/dataset/ucf_18_hajj/patch_256/'
version = 'v5.1'
dm_loss = 'msep'
lm_loss = 'bcep'
dm_weight = 1
lm_weight = 10
optimizer = 'adam'
INPUT_SIZE = 256
DM_FACTOR = 8  # 256/32
batch_size = 16

postfix = str(dm_weight) + '_' + dm_loss + '_' + str(lm_weight) + '_' + lm_loss
NAME = 'multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + \
       '_' + str(DM_FACTOR) + '_' + str(batch_size) + '_' + bench_name

data = pd.read_csv(bench_name + '_dm_' + str(INPUT_SIZE) + '.csv')
data['file_name'] = data['file_name'].apply(lambda name: dataset_path + name)
data['dm_name'] = data['dm_name'].apply(lambda name: dataset_path + name)
data['lm_name'] = data['lm_name'].apply(lambda name: dataset_path + name)

(train, valid) = train_test_split(data, test_size=0.25, random_state=42)

# TODO: data augmentation (brightness, rotate)
train_generator = DataGenerator(dataframe=train, bench_name=bench_name, batch_size=batch_size, im_size=INPUT_SIZE,
                                dm_factor=DM_FACTOR, mode='train', is_multi=True)

valid_generator = DataGenerator(dataframe=valid, bench_name=bench_name, batch_size=batch_size, im_size=INPUT_SIZE,
                                dm_factor=DM_FACTOR, mode='valid', is_multi=True)

if not resume:
    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    ##############################################################

    # 256
    model = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(inputs)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    # 128
    model = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    # 64
    model = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    # 32
    model = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    model = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    point_map_input = model

    # 32
    model = Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=(2, 2), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(2, 2), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), dilation_rate=(2, 2), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    density_map_output = Conv2D(filters=1, kernel_size=(1, 1), activation="relu",
                                padding="same", name='density_map_output')(model)

    # 32
    model = UpSampling2D(size=(2, 2))(point_map_input)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    # 64
    model = UpSampling2D(size=(2, 2))(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    # 128
    model = UpSampling2D(size=(2, 2))(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    # 256
    point_map_output = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid",
                              padding="same", name='point_map_output')(model)
    ##############################################################

    # construct the CNN
    model = Model(inputs, [density_map_output, point_map_output])

    # model.summary()
    plot_model(model, to_file="{}_model.png".format(NAME), show_shapes=True)

    model_json = model.to_json()
    with open("{}_model.json".format(NAME), 'w') as json_file:
        json_file.write(model_json)

else:
    with open("{}_model.json".format(NAME)) as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("{}_temp.weights".format(NAME))

loss = {
    'density_map_output': loss_index(dm_loss),
    'point_map_output': loss_index(lm_loss)
}

loss_weights = {
    'density_map_output': dm_weight,
    'point_map_output': lm_weight
}

metrics = {
    'density_map_output': mean_absolute_error_count,
    'point_map_output': mean_absolute_error_count
}

monitor = 'val_density_map_output_mean_absolute_error_count'

model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer, metrics=metrics)

bestpoint = ModelCheckpoint("{}_best.weights".format(NAME), monitor=monitor, verbose=1,
                            save_best_only=True, save_weights_only=True)

checkpoint = ModelCheckpoint("{}_temp.weights".format(NAME), verbose=1,
                             save_weights_only=True, period=5)

csv_logger = CSVLogger(filename="{}_log.txt".format(NAME), append=True)

# train the model
history = model.fit(x=train_generator, validation_data=valid_generator,
                    epochs=num_epochs, callbacks=[bestpoint, checkpoint, csv_logger])

model.save("{}.model".format(NAME), save_format='h5')

with open("{}.hist".format(NAME), 'wb') as hist:
    pickle.dump(history.history, hist)
