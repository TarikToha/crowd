import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model

from custom_dm_utility import *

resume = False
initial_epoch = 0  # n % 5 == 0
num_epochs = 20
bench_name = 'ucf_18v1'
# dataset_path = '/content/drive/My Drive/traffic_signal/dataset/shanghai/patch_512/'
# dataset_path = 'D:/dataset/shanghai/patch_256/'
dataset_path = '/home/user/dataset/ucf_18/patch_512/'
version = 'v7.3.1'
lm_loss = 'bcep'
optimizer = 'adam'
INPUT_SIZE = 512
batch_size = 4

postfix = lm_loss
NAME = 'multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + \
       '_' + str(batch_size) + '_' + bench_name

data = pd.read_csv(bench_name + '_train_' + str(INPUT_SIZE) + '.csv')
data['file_name'] = data['file_name'].apply(lambda name: dataset_path + name)
data['dm_name'] = data['dm_name'].apply(lambda name: dataset_path + name)
data['lm_name'] = data['lm_name'].apply(lambda name: dataset_path + name)

(train, valid) = train_test_split(data, test_size=0.25, random_state=42)

# TODO: data augmentation (brightness, rotate)
train_generator = DataGenerator(dataframe=train, bench_name=bench_name, batch_size=batch_size, im_size=INPUT_SIZE,
                                dm_factor=1, mode='train', scale=1, is_dm=False, is_lm=True)

valid_generator = DataGenerator(dataframe=valid, bench_name=bench_name, batch_size=batch_size, im_size=INPUT_SIZE,
                                dm_factor=1, mode='valid', scale=1, is_dm=False, is_lm=True)

init = GlorotUniform()

if not resume:
    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    ##############################################################

    # 256
    model = Conv2D(filters=16, kernel_size=(9, 9), activation='relu', kernel_initializer=init, padding="same")(inputs)
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=32, kernel_size=(7, 7), activation='relu', kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=16, kernel_size=(7, 7), activation='relu', kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = Conv2D(filters=8, kernel_size=(7, 7), activation='relu', kernel_initializer=init, padding="same")(model)
    model1 = BatchNormalization()(model)

    model = Conv2D(filters=20, kernel_size=(7, 7), activation='relu', kernel_initializer=init, padding="same")(inputs)
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=40, kernel_size=(5, 5), activation='relu', kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=20, kernel_size=(5, 5), activation='relu', kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = Conv2D(filters=10, kernel_size=(5, 5), activation='relu', kernel_initializer=init, padding="same")(model)
    model2 = BatchNormalization()(model)

    model = Conv2D(filters=24, kernel_size=(5, 5), activation='relu', kernel_initializer=init, padding="same")(inputs)
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=48, kernel_size=(3, 3), activation='relu', kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=24, kernel_size=(3, 3), activation='relu', kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = Conv2D(filters=12, kernel_size=(3, 3), activation='relu', kernel_initializer=init, padding="same")(model)
    model3 = BatchNormalization()(model)

    model = Concatenate()([model1, model2, model3])

    # 32
    model = UpSampling2D(size=(2, 2))(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)

    # 64
    model = UpSampling2D(size=(2, 2))(model)
    model = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)

    output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', kernel_initializer=init, padding="same")(model)
    ##############################################################

    # construct the CNN
    model = Model(inputs, output)

    # model = init_from_vgg(model, freeze=False)

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

metrics = localized_counting_error

monitor = 'val_localized_counting_error'

if optimizer == 'sgd':
    optimizer = SGD(lr=1e-3, momentum=0.95, decay=5e-4)

model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

bestpoint = ModelCheckpoint("{}_best.weights".format(NAME), monitor=monitor, verbose=1,
                            save_best_only=True, save_weights_only=True)

checkpoint = ModelCheckpoint("{}_temp.weights".format(NAME), verbose=1,
                             save_weights_only=True, period=5)

csv_logger = CSVLogger(filename="{}_log.txt".format(NAME), append=True)

stop_nan = TerminateOnNaN()

if not resume:
    initial_epoch = 0

# train the model
history = model.fit(x=train_generator, validation_data=valid_generator, initial_epoch=initial_epoch,
                    epochs=num_epochs, callbacks=[bestpoint, checkpoint, csv_logger, stop_nan])

model.save("{}.model".format(NAME), save_format='h5')

with open("{}.hist".format(NAME), 'wb') as hist:
    pickle.dump(history.history, hist)
