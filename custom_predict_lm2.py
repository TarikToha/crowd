import time

import pandas as pd
from tensorflow.keras import models

from conv_net_utility import *

is_parallel = True
is_sota = True
partial = False
bench_name = 'ucf_18v1'
version = 'v8.2.1'
lm_loss = 'bcep'
optimizer = 'adam'
INPUT_SIZE = 512
batch_size = 4

postfix = lm_loss
model_path = 'googledrive/aws/multi_' + version + '_' + postfix + '_' + optimizer + '_' + \
             str(INPUT_SIZE) + '_' + str(batch_size) + '_' + bench_name

img_path = 'dataset/ucf_18/ucf_18_test/'
np_path = 'dataset/ucf_18/patch_' + str(INPUT_SIZE) + '/'
idx_path = 'ucf_18v2_test.csv'

num_dir = 16
threshold1 = 5
# threshold2 = 40
conf_score1 = 0.5
conf_score2 = 0.75

if partial:
    model1 = models.load_model(model_path + '_temp.model', compile=False)
else:
    model1 = models.load_model(model_path + '.model', compile=False)

model2 = models.load_model(model_path + '_best.model', compile=False)

data = pd.read_csv(idx_path)
data['file_name'] = data['file_name'].apply(lambda name: img_path + name)


def batch_compute_precision_recall(parts_lm, prediction, threshold, conf_score, is_sota, i, count_pr):
    count, pr = compute_precision_recall(y_true=parts_lm, y_pred=prediction, threshold=threshold,
                                         conf_score=conf_score, is_sota=is_sota)
    count_pr[i] = [count, pr[0], pr[1]]


count_out1 = []
count_out2 = []
thread_list = []
count_pr = np.zeros((4, 3))
for idx, row in data.iterrows():
    file_name = row['file_name']
    # file_name = img_path + 'test_img_0260.jpg'
    start = time.time()
    _, parts, _, parts_lm = density_map_patches(file_name=file_name, INPUT_SIZE=INPUT_SIZE, dm_factor=1,
                                             scale=1, np_path=np_path, num_dir=num_dir, bench_name=bench_name)
    com1 = time.time()
    prediction1 = predict_multi_dm(img_array=parts, model=model1, is_multi=False, is_sota=is_sota)
    end1 = time.time()
    # count1 = np.sum(prediction[0])
    # count1 = np.sum(prediction[1])

    # print(parts.shape, parts_dm.shape, parts_lm.shape, prediction[0].shape, prediction[1].shape)
    # res1 = compute_psnr_ssim(y_true=parts_dm, y_pred=prediction[0])

    if not is_parallel:
        count11, pr11 = compute_precision_recall(y_true=parts_lm, y_pred=prediction1, threshold=threshold1,
                                                 conf_score=conf_score1, is_sota=is_sota)
        count12, pr12 = compute_precision_recall(y_true=parts_lm, y_pred=prediction1, threshold=threshold1,
                                                 conf_score=conf_score2, is_sota=is_sota)
        count_pr[0] = [count11, pr11[0], pr11[1]]
        count_pr[1] = [count12, pr12[0], pr12[1]]
    else:
        t = Thread(target=batch_compute_precision_recall,
                   args=(parts_lm, prediction1, threshold1, conf_score1, is_sota, 0, count_pr))
        t.start()
        thread_list.append(t)

        t = Thread(target=batch_compute_precision_recall,
                   args=(parts_lm, prediction1, threshold1, conf_score2, is_sota, 1, count_pr))
        t.start()
        thread_list.append(t)

    com2 = time.time()
    prediction2 = predict_multi_dm(img_array=parts, model=model2, is_multi=False, is_sota=is_sota)
    end2 = time.time()
    # count2 = np.sum(prediction[0])
    # count2 = np.sum(prediction[1])

    # res2 = compute_psnr_ssim(y_true=parts_dm, y_pred=prediction[0])
    if not is_parallel:
        count21, pr21 = compute_precision_recall(y_true=parts_lm, y_pred=prediction2, threshold=threshold1,
                                                 conf_score=conf_score1, is_sota=is_sota)
        count22, pr22 = compute_precision_recall(y_true=parts_lm, y_pred=prediction2, threshold=threshold1,
                                                 conf_score=conf_score2, is_sota=is_sota)
        count_pr[2] = [count21, pr21[0], pr21[1]]
        count_pr[3] = [count22, pr22[0], pr22[1]]
    else:
        t = Thread(target=batch_compute_precision_recall,
                   args=(parts_lm, prediction2, threshold1, conf_score1, is_sota, 2, count_pr))
        t.start()
        thread_list.append(t)

        t = Thread(target=batch_compute_precision_recall,
                   args=(parts_lm, prediction2, threshold1, conf_score2, is_sota, 3, count_pr))
        t.start()
        thread_list.append(t)

    for t in thread_list:
        t.join()

    out1 = file_name.split('/')[-1] + ',' + str(count_pr[0][0]) + ',' + str(count_pr[0][1]) + ',' + str(count_pr[0][2]) + ',' + \
           str(end1 - start) + ','

    out1 += str(count_pr[2][0]) + ',' + str(count_pr[2][1]) + ',' + str(count_pr[2][2]) + ',' + str(end2 - com2 + com1 - start)

    count_out1.append(out1)
    print(idx, out1)

    out2 = file_name.split('/')[-1] + ',' + str(count_pr[1][0]) + ',' + str(count_pr[1][1]) + ',' + str(count_pr[1][2]) + ',' + \
           str(end1 - start) + ','

    out2 += str(count_pr[3][0]) + ',' + str(count_pr[3][1]) + ',' + str(count_pr[3][2]) + ',' + str(end2 - com2 + com1 - start)

    count_out2.append(out2)
    print(idx, out2)
#     break

postfix = '_out.csv'
if is_sota:
    postfix = '_sota' + postfix

with open(model_path + '_' + str(threshold1) + '_' + str(conf_score1) + postfix, 'w') as csv_file:
    for out1 in count_out1:
        csv_file.write(out1 + '\n')

with open(model_path + '_' + str(threshold1) + '_' + str(conf_score2) + postfix, 'w') as csv_file:
    for out2 in count_out2:
        csv_file.write(out2 + '\n')
