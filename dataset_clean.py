import random

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

random.seed(123)

data = pd.read_csv('D:/Dropbox/academic/code/crowd/benchmark/nwpu_v1_train_512_full.csv')


def histogram():
    count = data['count']
    plt.hist(count, bins=20, range=(0, 200))
    plt.show()


clean_data = []
for idx, row in tqdm(data.iterrows()):
    file_name = row['file_name']
    # if 'train' in row['file_name']:
    if row['count'] == 0 and round(random.uniform(0, 0.52)) != 1:
        continue
    elif (1 <= row['count'] < 12) and round(random.uniform(0, 0.55)) != 1:
        continue
    elif (12 <= row['count'] < 29) and round(random.uniform(0, 0.61)) != 1:
        continue
    elif (29 <= row['count'] < 50) and round(random.uniform(0, 0.75)) != 1:
        continue
    elif (50 <= row['count'] < 76) and round(random.uniform(0, 0.9)) != 1:
        continue
    elif (76 <= row['count'] < 108) and round(random.uniform(0, 0.99)) != 1:
        continue

    # if 'test' in file_name:
    clean_data.append(row)

histogram()
print(len(clean_data))
data = pd.DataFrame(clean_data)
data.to_csv('df_labels_clean.csv', index=False)


