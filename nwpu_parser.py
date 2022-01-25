import numpy as np
import pandas as pd
from scipy import io
from tqdm import tqdm

base_path = 'D:/odrive/grad/traffic_signal/data/NWPU-Crowd/images/'
idx_file = 'D:/odrive/grad/traffic_signal/data/NWPU-Crowd/val.txt'
out_path = 'D:/dataset/nwpu_v1/'


def generate_txt_file(file_path, bbox):
    file_id = file_path.split('/')[-1].replace('.jpg', '')

    out = [file_id, str(bbox.shape[0])]
    for b in bbox:
        xmin, ymin, xmax, ymax = b[0], b[1], b[2], b[3]
        x, y = int((xmax + xmin) / 2), int((ymax + ymin) / 2)

        w, h = abs(xmax - xmin), abs(ymax - ymin)
        sigma_s = int(min(w, h))
        sigma_l = int(np.sqrt(w * w + h * h))
        level = int(np.floor(np.log10(w * h)))

        out.append(str(x))
        out.append(str(y))
        out.append(str(sigma_s))
        out.append(str(sigma_l))
        out.append(str(level))

    return out


data = pd.read_csv(idx_file, sep=' ', header=None)
data[0] = data[0].apply(lambda name: base_path + str(name) + '.jpg')

out_list = []
for idx, row in tqdm(data.iterrows()):
    img_file_name = row[0]
    # img_file_name = base_path + '3111.jpg'
    if '4k' in img_file_name:
        continue

    mat_file_name = img_file_name.replace('images', 'mats').replace('jpg', 'mat')
    mat = io.loadmat(mat_file_name)
    # annPoints = mat['annPoints']
    annBoxes = mat['annBoxes']

    # out_file_name = mat_file_name.split('/')[-1].replace('.mat', '')
    # np.save(out_path + out_file_name, annPoints)
    # print(out_file_name, annBoxes.shape)

    out = generate_txt_file(img_file_name, annBoxes)
    out = ' '.join(out)
    print(out)

    out_list.append(out)
    # break

out_list = pd.DataFrame(out_list)
out_list.to_csv('gt_out.txt', header=False, index=False)
