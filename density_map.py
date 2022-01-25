from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from tqdm import tqdm

base_path = 'D:/dataset/ucf_18v2/images/'
idx_path = 'D:/Dropbox/academic/code/crowd/benchmark/ucf_18v2_test.csv'

bench_name = 'ucf_18v2'

INPUT_SIZE = 512
num_dir = 16


def generate_density(pt, map_shape, gt_count, distance, density_map):
    d_map = np.zeros(map_shape, dtype=np.float32)
    d_map[pt[1], pt[0]] = 1

    if gt_count > 3:
        sigma = np.mean(distance[1:4]) * 0.3
    else:
        sigma = 15  # less than 4 persons in a patch

    d_map = gaussian_filter(d_map, sigma=sigma, mode='constant')

    density_map += d_map


def gaussian_filter_density(pts, map_shape):
    density_map = np.zeros(map_shape, dtype=np.float32)
    point_map = np.zeros(map_shape, dtype=np.float32)

    gt_count = pts.shape[0]

    if gt_count == 0:
        return density_map, point_map

    # tree = KDTree(pts, leafsize=64)
    # distances = tree.query(pts, k=4)[0]

    # thread_list = []
    # for i, pt in enumerate(pts):
    #     t = Thread(target=generate_density, args=(pt, map_shape, gt_count, distances[i], density_map))
    #     t.start()
    #     thread_list.append(t)
    #
    # for t in thread_list:
    #     t.join()

    point_map[pts[:, 1], pts[:, 0]] = 1

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    point_map = cv2.filter2D(src=point_map, ddepth=-1, kernel=kernel)
    point_map[point_map > 0] = 1

    return density_map, point_map


def gt_patch(np_mat, xmin, xmax, ymin, ymax):
    gt_list = []
    for val in np_mat:
        x = int(val[0])
        y = int(val[1])
        if (xmin <= x < xmax) and (ymin <= y < ymax):
            pt = [x - xmin, y - ymin]
            gt_list.append(pt)

    gt_list = np.array(gt_list)
    return gt_list


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


def gt_batch_dm(img_path, img, gt_list, xmin, xmax, ymin, ymax, mat_path, lm_path):
    cp_im = img[ymin:ymax, xmin:xmax]
    map_shape = cp_im.shape[0:2]
    cv2.imwrite(img_path, cp_im)

    density_map, point_map = gaussian_filter_density(gt_list, map_shape)
    # np.save(mat_path, density_map)
    np.save(lm_path, point_map)


def image_patches_dm(img_file_name, mat_file_name, INPUT_SIZE):
    img = cv2.imread(img_file_name)
    (h, w, c) = img.shape

    bottom = int(np.ceil(h / INPUT_SIZE)) * INPUT_SIZE - h
    right = int(np.ceil(w / INPUT_SIZE)) * INPUT_SIZE - w

    img = cv2.copyMakeBorder(img, top=0, bottom=bottom, left=0,
                             right=right, borderType=cv2.BORDER_CONSTANT)
    (h, w, c) = img.shape

    if bench_name == 'shanghai':
        mat = io.loadmat(mat_file_name)
        gt = np.array(mat['image_info'])[0, 0][0, 0][0]
    elif bench_name == 'ucf_18':
        mat = io.loadmat(mat_file_name)
        gt = np.array(mat['annPoints'])
    else:
        gt = np.load(mat_file_name)

    gt = validate_gt(gt, (h, w))

    dm_list = []
    thread_list = []
    count = 0
    for y in range(0, h, INPUT_SIZE):
        for x in range(0, w, INPUT_SIZE):
            xmin = x
            ymin = y
            xmax = xmin + INPUT_SIZE
            ymax = ymin + INPUT_SIZE
            # print(xmin, ymin, xmax, ymax)
            patch_id = count % num_dir
            count = count + 1

            gt_list = gt_patch(gt, xmin, xmax, ymin, ymax)

            img_path = img_file_name.replace('images', 'patch_im' + str(patch_id)) \
                .replace('.jpg', '_' + str(INPUT_SIZE) + '_' + str(count) + '.jpg')

            if bench_name == 'shanghai' or bench_name == 'ucf_18':
                mat_path = mat_file_name.replace('ground_truth', 'patch_dm' + str(patch_id)). \
                    replace('.mat', '_' + str(INPUT_SIZE) + '_' + str(count))
            else:
                mat_path = mat_file_name.replace('ground_truth', 'patch_dm' + str(patch_id)). \
                    replace('.npy', '_' + str(INPUT_SIZE) + '_' + str(count))

            lm_path = mat_path.replace('patch_dm', 'patch_lm')

            t = Thread(target=gt_batch_dm, args=(img_path, img, gt_list, xmin, xmax, ymin, ymax, mat_path, lm_path))
            t.start()
            thread_list.append(t)

            img_path = 'patch_im' + str(patch_id) + '/' + img_path.split('/')[-1]
            mat_path = 'patch_dm' + str(patch_id) + '/' + mat_path.split('/')[-1] + '.npy'
            lm_path = 'patch_lm' + str(patch_id) + '/' + lm_path.split('/')[-1] + '.npy'
            gt_count = gt_list.shape[0]

            dm_list.append([img_path, mat_path, lm_path, gt_count])

    for t in thread_list:
        t.join()

    return dm_list


def show_map(file_name):
    dm_lm = np.load(file_name)
    # print(dm_lm.max(), dm_lm.min())
    plt.imshow(dm_lm)
    plt.show()


data = pd.read_csv(idx_path)
data['file_name'] = data['file_name'].apply(lambda name: base_path + name)

dm_list_all = []
for idx, row in tqdm(data.iterrows()):
    img_file_name = row['file_name']
    if 'ucf_18v2' in bench_name and '_v2.' not in img_file_name:
        continue

    if bench_name == 'shanghai':
        mat_file_name = img_file_name.replace('images', 'ground_truth').replace('IMG', 'GT_IMG').replace('jpg', 'mat')
    elif bench_name == 'ucf_18':
        mat_file_name = img_file_name.replace('images', 'ground_truth').replace('.jpg', '_ann.mat')
    elif bench_name == 'nwpu' or bench_name == 'nwpu_v1':
        mat_file_name = img_file_name.replace('images', 'ground_truth').replace('jpg', 'npy')
    else:
        mat_file_name = img_file_name.replace('images', 'ground_truth').replace('_v2.jpg', '_ann_v2.npy')

    # print(idx, img_file_name, mat_file_name)

    dm_list = image_patches_dm(img_file_name, mat_file_name, INPUT_SIZE)
    dm_list_all += dm_list
    # break

data = pd.DataFrame(dm_list_all, columns=['file_name', 'dm_name', 'lm_name', 'count'])
data.to_csv(bench_name + '_dm_' + str(INPUT_SIZE) + '.csv', index=False)

# show_map('D:/dataset/nwpu/patch_lm10/0008_512_11.npy')
