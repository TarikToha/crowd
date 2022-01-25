import time

import pandas as pd
from tensorflow.keras import models

from conv_net_utility import *

# base_path = ''
base_path = 'D:/odrive/grad/traffic_signal/code/aws/'
# multi_v8.2.5_bcep_adam_512_4_ucf_18v1_best
is_sota = False
is_best = True
bench_name = 'ucf_18v1'
version = 'v8.2.5'
lm_loss = 'bcep'
optimizer = 'adam'
INPUT_SIZE = 512
batch_size = 4

postfix = lm_loss
model_path = base_path + 'multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + '_' + \
             str(batch_size) + '_' + bench_name

ext = ''
if is_best:
    ext += '_best'
ext += '.model'

model = models.load_model(model_path + ext, compile=False)

# model.summary()

img_path = 'test_img_0215.jpg'
prefix = 'D:/odrive/grad/traffic_signal/dataset/ucf_18/ucf_18_test/'
# prefix = '/content/drive/My Drive/traffic_signal/dataset/ucf_18/ucf_18_test/'
np_path = 'D:/odrive/grad/traffic_signal/dataset/ucf_18/patch_' + str(INPUT_SIZE) + '/'
# np_path = '/content/drive/My Drive/traffic_signal/dataset/ucf_18/patch_' + str(INPUT_SIZE) + '/'
num_dir = 16

im_id = 4
layer_id = 164

grid, parts, _, parts_lm = density_map_patches(file_name=prefix + img_path, INPUT_SIZE=INPUT_SIZE, dm_factor=1,
                                               scale=1, np_path=np_path, num_dir=num_dir, bench_name=bench_name)
# print(parts.shape, parts_lm.shape, grid)
im = parts[im_id]
im = np.expand_dims(im, axis=0)
# print(im.shape)

prediction = predict_layers(img_array=im, model=model, layer_id=layer_id)
print(layer_id, prediction.shape)

im = np.squeeze(im, axis=0)
prediction = np.squeeze(prediction, axis=0)
# print(im.shape)
visualize_map(y_img=im, y_lm=parts_lm[im_id], y_pred=prediction,
              INPUT_SIZE=INPUT_SIZE, bench_name=bench_name, layer_id=layer_id)
