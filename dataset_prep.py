import random

import pandas as pd
from tqdm import tqdm

data = pd.read_csv('/home/toha/Dropbox/academic/code/test/benchmark/ucf_18_all.csv')

clean_data = []
for idx, row in tqdm(data.iterrows()):
    # if row['count'] == 0 and round(random.uniform(0, 0.55)) != 1:
    #     continue
    # elif (1 <= row['count'] < 11) and round(random.uniform(0, 0.6)) != 1:
    #     continue
    #     #         ((4 <= row['count'] < 9) and round(random.uniform(0, 0.8)) != 1):

    if 'test' in row['file_name']:
        clean_data.append(row)

data = pd.DataFrame(clean_data)
data.to_csv('df_labels_clean.csv', index=False)
# print(data)
