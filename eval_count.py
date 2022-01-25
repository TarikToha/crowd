import pandas as pd
import numpy as np

base_loc = 'D:/odrive/grad/traffic_signal/code/aws/'

is_sota = True
bench_name = 'shanghai_B'
version = ['v8.2']
lm_loss = 'bcep'
optimizer = 'adam'
INPUT_SIZE = 512
batch_size = 4
threshold = [20]
conf_score = 0.5

prefix = base_loc + 'multi_'
postfix = '_' + lm_loss + '_' + optimizer + '_' + str(INPUT_SIZE) + '_' + str(batch_size) + '_' + bench_name + '_'

post_ext = '_out.csv'
if is_sota:
    post_ext = '_sota' + post_ext

out = []
for v in version:
    for i in range(1, 6):
        ver = v + '.' + str(i)
        model_path = prefix + ver + postfix
        for thresh in threshold:
            idx_file = model_path + str(thresh) + '_' + str(conf_score) + post_ext

            data = pd.read_csv(idx_file, header=None)
            count = data[5]
            out.append(count)


out = np.array(out)
out = np.transpose(out)

out = pd.DataFrame(out, columns=None)
out.to_csv('eval_summary.csv', index=False)
