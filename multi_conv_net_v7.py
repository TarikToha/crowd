import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, MaxPooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model

from custom_dm_utility import *

resume = False
initial_epoch = 0  # n % 5 == 0
num_epochs = 100
bench_name = 'shanghai_A'
# dataset_path = '/content/drive/My Drive/traffic_signal/dataset/shanghai/patch_256/'
dataset_path = 'D:/odrive/grad/traffic_signal/dataset/shanghai/patch_256/'
# dataset_path = 'dataset/ucf_18/patch_256/'
version = 'v8.2'
lm_loss = 'bcep2'
optimizer = 'adam'
INPUT_SIZE = 256
batch_size = 8
seed = 42

postfix = lm_loss
NAME = 'multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + \
       '_' + str(batch_size) + '_' + bench_name

data = pd.read_csv(bench_name + '_dm_' + str(INPUT_SIZE) + '.csv')
data['file_name'] = data['file_name'].apply(lambda name: dataset_path + name)
data['dm_name'] = data['dm_name'].apply(lambda name: dataset_path + name)
data['lm_name'] = data['lm_name'].apply(lambda name: dataset_path + name)

(train, valid) = train_test_split(data, test_size=0.25, random_state=42)

# TODO: data augmentation (brightness, rotate)
train_generator = DataGenerator(dataframe=train, bench_name=bench_name, batch_size=batch_size, im_size=INPUT_SIZE,
                                dm_factor=1, mode='train', scale=1, is_dm=False, is_lm=True)

valid_generator = DataGenerator(dataframe=valid, bench_name=bench_name, batch_size=batch_size, im_size=INPUT_SIZE,
                                dm_factor=1, mode='valid', scale=1, is_dm=False, is_lm=True)

# init = HeUniform(seed=seed)
# init = RandomNormal(stddev=0.01)
init = GlorotUniform(seed=seed)

if not resume:
    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    ##############################################################

    # 256
    model = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer=init, padding="same")(inputs)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    # 128
    model = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    route = model
    model = Conv2D(filters=32, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])
    model = MaxPooling2D(pool_size=(2, 2))(model)

    # 64
    model = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    route = model
    model = Conv2D(filters=64, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    route = model
    model = Conv2D(filters=64, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])
    model = MaxPooling2D(pool_size=(2, 2))(model)

    # 32
    model = Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    route = model
    model = Conv2D(filters=128, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    route = model
    model = Conv2D(filters=128, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    route = model
    model = Conv2D(filters=128, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    route = model
    model = Conv2D(filters=128, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    model = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    route = model
    model = Conv2D(filters=128, kernel_size=(1, 1), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    route = model
    model = Conv2D(filters=128, kernel_size=(1, 1), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    route = model
    model = Conv2D(filters=128, kernel_size=(1, 1), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    route = model
    model = Conv2D(filters=128, kernel_size=(1, 1), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    # 32
    model = UpSampling2D(size=(2, 2))(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    route = model
    model = Conv2D(filters=64, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    model = Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    # 64
    model = UpSampling2D(size=(2, 2))(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    route = model
    model = Conv2D(filters=32, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    model = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    # 128
    model = UpSampling2D(size=(2, 2))(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    route = model
    model = Conv2D(filters=16, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Add()([model, route])

    model = Conv2D(filters=32, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=16, kernel_size=(1, 1), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), dilation_rate=(2, 2), kernel_initializer=init,
                   padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    # 256
    model = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(model)
    model = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(model)
    # model = conv_attention_module(model, init)

    model = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=16, kernel_size=(1, 1), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer=init, padding="same")(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', kernel_initializer=init,
                    padding="same")(model)
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
