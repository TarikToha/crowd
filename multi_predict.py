import time

import pandas as pd
from keras import models

from conv_net_utility import *

bench_name = 'ucf_18_dense'

reg_cls_model_name = 'history/custom_cnn_reg_10cls_' + bench_name + '.model'

img_path = '/media/user/Data/odrive/grad/traffic_signal/code/ucf_18/images/'
idx_path = 'benchmark/' + bench_name + '_test.csv'

PATCH_SIZE = 256
INPUT_SIZE = 128

net = models.load_model(reg_cls_model_name)

data = pd.read_csv(idx_path)
data['file_name'] = data['file_name'].apply(lambda name: img_path + name)

reg_cls_out = []

for idx, row in data.iterrows():
    file_name = row['file_name']
    start = time.time()
    parts = image_patches(file_name=file_name, PATCH_SIZE=PATCH_SIZE, INPUT_SIZE=INPUT_SIZE)
    prediction = predict(img_array=parts, model=net)
    count_reg = np.sum(prediction[0])
    count_cls = class_to_count(prediction=prediction[1], bench_name=bench_name)
    end = time.time()
    out = file_name.split('/')[-1] + ',' + str(count_reg) + ',' + str(count_cls) + ',' + str(end - start)
    reg_cls_out.append(out)
    print(idx, out)

with open('multi_out.csv', 'w') as csv_file:
    for out in reg_cls_out:
        csv_file.write(out + '\n')
