from threading import Thread

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

base_loc = 'D:/dataset/nwpu_v1/images/'
out_file = 'mean_std_nwpu_v1_512'
idx_file = 'D:/Dropbox/academic/code/crowd/benchmark/nwpu_v1_all.csv'


def batch_read(row, mean_std, i):
    file_name = row['file_name']
    img = cv2.imread(file_name)
    print(file_name.split('/')[-1], img.shape, row['count'])

    # img = img / 255
    # mean, std = cv2.meanStdDev(img)
    # mean = np.squeeze(mean, axis=-1)
    # std = np.squeeze(std, axis=-1)
    # mean_std[i] = np.concatenate((mean, std))


data = pd.read_csv(idx_file)
data['file_name'] = data['file_name'].apply(lambda name: base_loc + name)

mean_std = np.zeros((len(data), 6))
thread_list = []
for idx, row in (data.iterrows()):
    batch_read(row, mean_std, idx)
    # t = Thread(target=batch_read, args=(row, mean_std, idx))
    # t.start()
    # thread_list.append(t)

    # if idx % 100 == 0:
    #     for t in thread_list:
    #         t.join()

    # break

# for t in thread_list:
#     t.join()

# np.save(out_file, mean_std)

# 512
# [0.30298058 0.31315339 0.34048589 0.18599708 0.18967585 0.20333949] NWPUV1
# [0.33574451 0.3482778  0.37829334 0.20596884 0.2087179  0.22302162] NWPU
# [0.28796011 0.29466529 0.32114294 0.18639777 0.18898598 0.20312717] UCF18V1
# [0.30767617 0.31538381 0.34562042 0.18573873 0.18862324 0.20404656] UCF18
# [0.23139508 0.2374646  0.26217736 0.22035383 0.22130816 0.23672004] SHHA
# [0.32018929 0.33140811 0.3346443  0.23334228 0.23878472 0.24407709] SHHB

# 256
# [0.33024199 0.33794826 0.36828972 0.1767343  0.18027293 0.19388486] UCF18V1
# [0.33189618 0.34023062 0.3728201  0.16668856 0.17049365 0.18432766] UCF18
# [0.2983559  0.3062046  0.33802828 0.21958432 0.22060138 0.23461229] SHHA
# [0.42692594 0.44189072 0.44618444 0.18650701 0.19225413 0.19943645] SHHB

# 128
# [0.32242125 0.3308784  0.36531266 0.20520084 0.20671864 0.21981789] SHHA
# [0.42691906 0.44187748 0.4461924  0.16749409 0.17348416 0.18042755] SHHB

# mean_std = np.load(out_file + '.npy')
# print(mean_std.shape)
# mean_std = np.mean(mean_std, axis=0)
# print(mean_std)
