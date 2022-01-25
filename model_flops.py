from keras_flops import get_flops
import os
from tensorflow.keras import models

model_path = 'D:/Dropbox/shared/traffic_signal/multi_v7.3.1_bcep_adam_512_4_ucf_18v1_best'
# model_path = 'D:/odrive/grad/traffic_signal/code/aws/multi_v6.3.1_bcep_adam_512_4_ucf_18v1_best'
keras_model = models.load_model(model_path + '.model', compile=False)
keras_model.summary()

flops = get_flops(keras_model)
print(flops/1e9)

mb = os.stat(model_path + '.model').st_size
print(mb/1e6)