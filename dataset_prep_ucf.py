import pandas as pd

data = pd.read_csv('ucf_13_labels.csv')
# data = data.drop(columns=['class_num', 'class_alpha'])

clean_data = []

# root_img = []
# for i in range(50):
#     root_img.append(str(i + 1))
#
# X = np.array(root_img)
# kf = KFold(n_splits=5, shuffle=True)
# splits = kf.split(X)
#
# for idx, (train_index, test_index) in enumerate(splits):
#     train_split = []
#     tr = X[train_index]
#     for t in tr:
#         reg = 'patch/' + str(t) + '_'
#         match = data[data['file_name'].str.contains(reg, regex=True)]
#         train_split.append(match)
#
#     train_split = pd.concat(train_split)
#     train_split = pd.DataFrame(train_split)
#     train_split.to_csv('train_' + str(idx) + '.csv', index=False)
#
#     test_split = []
#     ts = X[test_index]
#     for t in ts:
#         reg = 'patch/' + str(t) + '_'
#         match = data[data['file_name'].str.contains(reg, regex=True)]
#         test_split.append(match)
#
#     test_split = pd.concat(test_split)
#     test_split = pd.DataFrame(test_split)
#     test_split.to_csv('test_' + str(idx) + '.csv', index=False)
#
#     print(idx)

test_file = 'ucf_13_1_test.csv'

split = pd.read_csv(test_file)
count = []
for idx, row in split.iterrows():
    row = data[data['file_name'] == row['file_name']]
    count.append(row)

count = pd.concat(count)
count = pd.DataFrame(count)
count.to_csv(test_file, index=False)
print(count)
