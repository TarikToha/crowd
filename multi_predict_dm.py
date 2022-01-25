import time

import pandas as pd
from tensorflow.keras import models

from conv_net_utility import *

is_sota = False
partial = False
bench_name = 'shanghai_A'
version = 'v8.2'
dm_loss = 'msep'
lm_loss = 'bcep'
dm_weight = 1
lm_weight = 1
optimizer = 'adam'
INPUT_SIZE = 128
DM_FACTOR = 8  # 256/32
batch_size = 16

postfix = str(dm_weight) + '_' + dm_loss + '_' + str(lm_weight) + '_' + lm_loss
model_path = 'multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + \
             '_' + str(DM_FACTOR) + '_' + str(batch_size) + '_' + bench_name

img_path = 'dataset/shanghai/shanghai_test/'
np_path = 'dataset/shanghai/patch_' + str(INPUT_SIZE) + '/'
idx_path = bench_name + '_test.csv'

num_dir = 16
threshold1 = 20
threshold2 = 40
conf_score = 0.5

if partial:
    model1 = models.load_model(model_path + '_temp.model', compile=False)
else:
    model1 = models.load_model(model_path + '.model', compile=False)

model2 = models.load_model(model_path + '_best.model', compile=False)

data = pd.read_csv(idx_path)
data['file_name'] = data['file_name'].apply(lambda name: img_path + name)

count_out1 = []
count_out2 = []
for idx, row in data.iterrows():
    file_name = row['file_name']
    # file_name = img_path + 'test_img_0260.jpg'
    start = time.time()
    parts, parts_dm, parts_lm = density_map_patches(file_name=file_name, INPUT_SIZE=INPUT_SIZE, dm_factor=DM_FACTOR,
                                                    scale=1, np_path=np_path, num_dir=num_dir, bench_name=bench_name)
    com = time.time()
    prediction = predict_multi_dm(img_array=parts, model=model1, is_multi=True, is_sota=is_sota)
    # count1 = np.sum(prediction[0])
    # count1 = np.sum(prediction[1])
    end1 = time.time()
    # print(parts.shape, parts_dm.shape, parts_lm.shape, prediction[0].shape, prediction[1].shape)
    res1 = compute_psnr_ssim(y_true=parts_dm, y_pred=prediction[0])
    count11, pr11 = compute_precision_recall(y_true=parts_lm, y_pred=prediction[1], threshold=threshold1,
                                             conf_score=conf_score, is_sota=is_sota)
    count12, pr12 = compute_precision_recall(y_true=parts_lm, y_pred=prediction[1], threshold=threshold2,
                                             conf_score=conf_score, is_sota=is_sota)

    prediction = predict_multi_dm(img_array=parts, model=model2, is_multi=True, is_sota=is_sota)
    # count2 = np.sum(prediction[0])
    # count2 = np.sum(prediction[1])
    end2 = time.time()
    res2 = compute_psnr_ssim(y_true=parts_dm, y_pred=prediction[0])
    count21, pr21 = compute_precision_recall(y_true=parts_lm, y_pred=prediction[1], threshold=threshold1,
                                             conf_score=conf_score, is_sota=is_sota)
    count22, pr22 = compute_precision_recall(y_true=parts_lm, y_pred=prediction[1], threshold=threshold2,
                                             conf_score=conf_score, is_sota=is_sota)

    out1 = file_name.split('/')[-1] + ',' + str(count11) + ',' + str(res1[0]) + ',' + \
           str(res1[1]) + ',' + str(pr11[0]) + ',' + str(pr11[1]) + ',' + str(end1 - start) + ','

    out1 += str(count21) + ',' + str(res2[0]) + ',' + str(res2[1]) + ',' + \
            str(pr21[0]) + ',' + str(pr21[1]) + ',' + str(end2 - end1 + com - start)

    count_out1.append(out1)
    print(idx, out1)

    out2 = file_name.split('/')[-1] + ',' + str(count12) + ',' + str(res1[0]) + ',' + \
           str(res1[1]) + ',' + str(pr12[0]) + ',' + str(pr12[1]) + ',' + str(end1 - start) + ','

    out2 += str(count22) + ',' + str(res2[0]) + ',' + str(res2[1]) + ',' + \
            str(pr22[0]) + ',' + str(pr22[1]) + ',' + str(end2 - end1 + com - start)

    count_out2.append(out2)
    print(idx, out2)
#     break

with open(model_path + '_' + str(threshold1) + '_out.csv', 'w') as csv_file:
    for out1 in count_out1:
        csv_file.write(out1 + '\n')

with open(model_path + '_' + str(threshold2) + '_out.csv', 'w') as csv_file:
    for out2 in count_out2:
        csv_file.write(out2 + '\n')
