from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io
from scipy.signal import windows
from scipy.spatial import KDTree
from tqdm import tqdm

base_path = 'D:/odrive/grad/traffic_signal/code/ucf_18/images/'
idx_path = 'D:/Dropbox/academic/code/test/benchmark/ucf_18v1_test.csv'

bench_name = 'ucf_18v1'
threshold = 20
conf_score = 0.5


def get_distances(data):
    tree = KDTree(data, leafsize=64)
    distances = tree.query(data, k=2)[0]
    distances = distances[:, 1]

    return distances


def get_kernel(kernlen=49, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = windows.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def gaussian_function(h, w, gt_point, pred_point, sigma):
    point_map = np.zeros((h, w), dtype=np.float32)
    point_map[gt_point[1], gt_point[0]] = 1

    kernel = get_kernel(std=sigma)
    point_map = cv2.filter2D(src=point_map, ddepth=-1, kernel=kernel)
    point_map /= point_map.max()
    val = point_map[pred_point[1], pred_point[0]]
    return val, point_map


def get_variance(h, w, data):
    tree = KDTree(data, leafsize=64)
    gt_idx = tree.query(data, k=2)[1]

    count = 0
    for idx in gt_idx:
        g = data[idx[0]]
        gt_point = g[0], g[1]

        p = data[idx[1]]
        pred_point = p[0], p[1]

        val, _ = gaussian_function(h=h, w=w, gt_point=gt_point, pred_point=pred_point, sigma=threshold)
        # print(val)
        if val >= conf_score:
            count += 1

    return gt_idx.shape[0], count


def validate_gt(np_mat, map_shape):
    gt_list = []
    for val in np_mat:
        x = int(val[0])
        y = int(val[1])
        if (x < map_shape[1]) and (y < map_shape[0]):
            pt = [x, y]
            gt_list.append(pt)

    gt_list = np.array(gt_list)
    return gt_list


def get_gt_list(h, w, mat_file_name):
    if 'shanghai' in bench_name:
        mat = io.loadmat(mat_file_name)
        gt = np.array(mat['image_info'])[0, 0][0, 0][0]
    else:
        if '_4k.' not in img_file_name:
            mat = io.loadmat(mat_file_name)
            gt = np.array(mat['annPoints'])
        else:
            gt = np.load(mat_file_name)

    gt = validate_gt(gt, (h, w))
    # dist = get_distances(gt)
    gt, count = get_variance(h, w, gt)

    # return dist
    return gt, count


def batch_variance(h, w, mat_file_name, i, gt_count):
    gt, count = get_gt_list(h, w, mat_file_name)

    gt_count[i] = [gt, count]


data = pd.read_csv(idx_path)
data['file_name'] = data['file_name'].apply(lambda name: base_path + name)

it = 0
thread_list = []
gt_count = np.zeros((data.size, 2))
dist_list_all = []
for idx, row in data.iterrows():
    img_file_name = row['file_name']
    # if 'v' in bench_name and '_4k.' not in img_file_name:
    #     continue

    if 'test_' not in img_file_name:
        continue

    img = cv2.imread(img_file_name)
    (h, w, c) = img.shape

    if 'shanghai' in bench_name:
        mat_file_name = img_file_name.replace('images', 'ground_truth').replace('IMG', 'GT_IMG').replace('jpg', 'mat')
    else:
        if '_4k.' not in img_file_name:
            mat_file_name = img_file_name.replace('images', 'ground_truth').replace('.jpg', '_ann.mat')
        else:
            mat_file_name = img_file_name.replace('images', 'ground_truth').replace('_4k.jpg', '_ann_4k.npy')

    # print(idx, img_file_name, mat_file_name)

    # gt = get_gt_list(mat_file_name)
    t = Thread(target=batch_variance, args=(h, w, mat_file_name, idx, gt_count))
    t.start()
    thread_list.append(t)

    # dist_list_all += list(gt)

    print(idx)
    it += 1
    if it % 20 == 0:
        for t in thread_list:
            t.join()
    # break

for t in thread_list:
    t.join()

data = pd.DataFrame(gt_count)
data.to_csv('gt_count_' + bench_name + '_' + str(threshold) + '_' + str(conf_score) + '.csv', index=False)

# distances = np.array(dist_list_all)
# plt.hist(distances, range=(0, 100))
# plt.show()
