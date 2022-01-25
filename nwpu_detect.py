import time
import pandas as pd
from tensorflow.keras import models
from tqdm import tqdm

from conv_net_utility import *

# base_path = ''
base_path = 'D:/odrive/grad/traffic_signal/code/aws/'
# multi_v8.2.5_bcep_adam_512_4_ucf_18v1_best
is_sota = False
bench_name = 'nwpu_v1'
version = 'v8.2.2'
lm_loss = 'bcep'
optimizer = 'adam'
INPUT_SIZE = 512
batch_size = 4

postfix = lm_loss
model_path = base_path + 'multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + '_' + \
             str(batch_size) + '_' + bench_name

model = models.load_model(model_path + '_best_60.model', compile=False)

img_path = 'D:/odrive/grad/traffic_signal/dataset/nwpu/nwpu_test/'
idx_path = 'benchmark/' + bench_name + '_test.csv'


def detect_frame(file_name):
    start = time.time()

    frame = cv2.imread(file_name)
    parts, grid = make_patch(frame, INPUT_SIZE, bench_name)
    parts_lm = predict_multi_dm(img_array=parts, model=model, is_multi=False, is_sota=is_sota)

    # print(parts.shape, parts_lm.shape, grid)

    count, _, lm_map = get_localization_map(parts=parts, pred_parts=parts_lm, grid=grid, is_gt=False, is_sota=is_sota)
    end = time.time()

    # print(idx, count, gt_img.shape, lm_map.shape, (end - start))

    return lm_map


def draw_frame(gt_img, lm_map):
    img = restore_image(gt_img, INPUT_SIZE, bench_name)
    frame = draw_output(img=img, lm_map=lm_map, lm=False, radius=5, color=(0, 0, 255), thickness=-1)
    return frame


def merge_gt_pred_img(gt_img, lm_map1, lm_map2):
    img = restore_image(gt_img, INPUT_SIZE, bench_name)

    frame = draw_output(img=img, lm_map=lm_map1, lm=False, radius=5, color=(0, 0, 255), thickness=-1)
    frame = draw_output(img=frame, lm_map=lm_map2, lm=False, radius=10, color=(255, 0, 0), thickness=2)

    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)

    return frame


def generate_txt_file(file_path, point_map):
    file_id = file_path.split('/')[-1].replace('.jpg', '')
    # print(file_id)

    lm_pts = np.nonzero(point_map)
    lm_pts = list(zip(lm_pts[1], lm_pts[0]))
    out = [file_id, str(len(lm_pts))]
    for pt in lm_pts:
        out.append(str(pt[0]))
        out.append(str(pt[1]))

    return out


def show_map(lm):
    plt.imshow(lm)
    plt.show()


data = pd.read_csv(idx_path)
data['file_name'] = data['file_name'].apply(lambda name: img_path + name)

out_list = []
for idx, row in tqdm(data.iterrows()):
    file_name = row['file_name']

    lm_map = detect_frame(file_name)
    # show_map(lm_map)
    out = generate_txt_file(file_name, lm_map)
    out = ' '.join(out)
    # print(out)

    out_list.append(out)
    # break

postfix = '_out.txt'
if is_sota:
    postfix = '_sota' + postfix

out_list = pd.DataFrame(out_list)
out_list.to_csv(model_path + postfix, header=False, index=False)
