import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model

from custom_dm_utility import *

resume = False
num_epochs = 50
# bench_name = 'shanghai_A'
bench_name = 'ucf_18'
# dataset_path = '/media/toha/Data/odrive/grad/traffic_signal/dataset/ucf_18_hajj/patch_256/'
# dataset_path = '/content/drive/My Drive/traffic_signal/dataset/shanghai/patch_256/'
# dataset_path = 'dataset/shanghai_A/patch_256/'
dataset_path = 'dataset/ucf_18/patch_256/'
version = 'v6'
lm_loss = 'bcep'
optimizer = 'sgd'
INPUT_SIZE = 256
DM_FACTOR = 8  # 256/32
batch_size = 32

postfix = lm_loss
NAME = 'multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + \
       '_' + str(DM_FACTOR) + '_' + str(batch_size) + '_' + bench_name

data = pd.read_csv(bench_name + '_dm_' + str(INPUT_SIZE) + '.csv')
data['file_name'] = data['file_name'].apply(lambda name: dataset_path + name)
data['dm_name'] = data['dm_name'].apply(lambda name: dataset_path + name)
data['lm_name'] = data['lm_name'].apply(lambda name: dataset_path + name)

(train, valid) = train_test_split(data, test_size=0.25, random_state=42)

# TODO: data augmentation (brightness, rotate)
train_generator = DataGenerator(dataframe=train, bench_name=bench_name, batch_size=batch_size, im_size=INPUT_SIZE,
                                dm_factor=DM_FACTOR, mode='train', scale=1, is_dm=False, is_lm=True)

valid_generator = DataGenerator(dataframe=valid, bench_name=bench_name, batch_size=batch_size, im_size=INPUT_SIZE,
                                dm_factor=DM_FACTOR, mode='valid', scale=1, is_dm=False, is_lm=True)


if not resume:
    input = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    ##############################################################

    # 256
    model = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(input)
    model = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    # 128
    model = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    # 64
    model = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same")(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same")(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same")(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    # 32
    model = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")(model)
    model = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")(model)
    model = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")(model)

    model = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation="relu",
                   padding="same")(model)
    model = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation="relu",
                   padding="same")(model)
    model = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation="relu",
                   padding="same")(model)

    # 32
    model = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), activation="relu", padding="same")(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), activation="relu", padding="same")(model)

    # 64
    model = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), activation="relu", padding="same")(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=(2, 2), activation="relu", padding="same")(model)

    # 128
    model = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu", padding="same")(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(2, 2), activation="relu", padding="same")(model)

    # 256
    output = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", padding="same")(model)
    ##############################################################

    # construct the CNN
    model = Model(input, output)

    model = init_from_vgg(model, freeze=True)

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

loss = loss_index(lm_loss)

metrics = mean_absolute_error_count

monitor = 'val_mean_absolute_error_count'

if optimizer == 'sgd':
    optimizer = SGD(lr=1e-5, momentum=0.95, decay=5e-4)

model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

bestpoint = ModelCheckpoint("{}_best.weights".format(NAME), monitor=monitor, verbose=1,
                            save_best_only=True, save_weights_only=True)

checkpoint = ModelCheckpoint("{}_temp.weights".format(NAME), verbose=1,
                             save_weights_only=True, period=5)

csv_logger = CSVLogger(filename="{}_log.txt".format(NAME), append=True)

stop_nan = TerminateOnNaN()

# train the model
history = model.fit(x=train_generator, validation_data=valid_generator,
                    epochs=num_epochs, callbacks=[bestpoint, checkpoint, csv_logger, stop_nan])

model.save("{}.model".format(NAME), save_format='h5')

with open("{}.hist".format(NAME), 'wb') as hist:
    pickle.dump(history.history, hist)
