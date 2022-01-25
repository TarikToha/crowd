import pandas as pd

full_path = 'D:/Dropbox/academic/code/test/benchmark/ucf_18_dm_512_full.csv'
uhd_path = 'D:/Dropbox/academic/code/test/benchmark/ucf_18v1_all.csv'

full_data = pd.read_csv(full_path)
uhd_data = pd.read_csv(uhd_path)

print(len(full_data))
for idx0, row0 in uhd_data.iterrows():
    file_name0 = row0['file_name']
    if '_4k' not in file_name0:
        continue
    file_name0 = file_name0.replace('.jpg', '').replace('_4k', '')

    file_name1 = full_data['file_name']
    dup = file_name1[file_name1.str.contains(file_name0, regex=False, na=False)]
    full_data.drop(dup.index, inplace=True)

    print(len(full_data))

data = pd.DataFrame(full_data)
data.to_csv('df_labels_clean.csv', index=False)
