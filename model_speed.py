import time

import pandas as pd
from tensorflow.keras import models

from conv_net_utility import *

is_parallel = True
is_sota = True
partial = False
bench_name = 'ucf_18v1'
version = 'v5.3.1'
lm_loss = 'bcep'
optimizer = 'adam'
INPUT_SIZE = 512
batch_size = 4

postfix = lm_loss
model_path = 'D:/odrive/grad/traffic_signal/code/aws/multi_' + version + '_' + postfix + '_' + optimizer + '_' + \
             str(INPUT_SIZE) + '_' + str(batch_size) + '_' + bench_name

img_path = 'D:/odrive/grad/traffic_signal/dataset/ucf_18/ucf_18_test/'
idx_path = 'benchmark/' + bench_name + '_test.csv'

model = models.load_model(model_path + '_best.model', compile=False)

data = pd.read_csv(idx_path)
data['file_name'] = data['file_name'].apply(lambda name: img_path + name)

count_out = []
for idx, row in data.iterrows():
    file_name = row['file_name']
    # file_name = img_path + 'test_img_0260.jpg'

    frame = cv2.imread(file_name)
    parts, _ = make_patch(frame, INPUT_SIZE, bench_name)

    start = time.time()
    prediction = predict_multi_dm(img_array=parts, model=model, is_multi=False, is_sota=is_sota)
    end = time.time()

    out = file_name.split('/')[-1] + ',' + str(end - start)

    count_out.append(out)
    print(idx, out)
#     break

postfix = '_time.csv'
if is_sota:
    postfix = '_sota' + postfix

with open(model_path + postfix, 'w') as csv_file:
    for out in count_out:
        csv_file.write(out + '\n')
