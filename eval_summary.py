import pandas as pd

base_loc = 'D:/odrive/grad/traffic_signal/code/aws/'

is_sota = True
bench_name = 'ucf_18v1'
version = ['v7.3']
lm_loss = 'bcep'
optimizer = 'adam'
INPUT_SIZE = 512
batch_size = 4
threshold = [20, 40]
conf_score = 0.5

prefix = base_loc + 'multi_'
postfix = '_' + lm_loss + '_' + optimizer + '_' + str(INPUT_SIZE) + '_' + str(batch_size) + '_' + bench_name + '_'

post_ext = '_out.csv'
if is_sota:
    post_ext = '_sota' + post_ext

columns = ['benchmark', 'model_name', 'precision_20', 'recall_20', 'precision_40', 'recall_40']
# columns = ['benchmark', 'model_name', 'precision_20', 'recall_20', 'mle_20', 'precision_40', 'recall_40', 'mle_40']
print(columns)

out = []
for v in version:
    for i in range(1, 6):
        ver = v + '.' + str(i)
        model_path = prefix + ver + postfix
        pr = [bench_name, ver]
        for thresh in threshold:
            idx_file = model_path + str(thresh) + '_' + str(conf_score) + post_ext

            data = pd.read_csv(idx_file, header=None)
            avg = data.mean(axis=0)
            avg = avg[5:7]
            # avg = avg[6:9]

            pr.extend(avg)

        out.append(pr)
        print(pr)

out = pd.DataFrame(out, columns=columns)
out.to_csv('eval_summary.csv', index=False)
