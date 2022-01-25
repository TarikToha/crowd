import cv2
import numpy as np
import pandas as pd
from scipy import io

base_path = 'D:/dataset/ucf_18v2/images/'
idx_path = 'D:/Dropbox/academic/code/crowd/benchmark/ucf_18_test.csv'
out_path = 'D:/dataset/ucf_18v2/'
# idx_path = 'resize_list.csv'

data = pd.read_csv(idx_path)
data['file_name'] = data['file_name'].apply(lambda name: base_path + name)

resolution = [[720, 1280], [1080, 1920], [2160, 3840]]
mp_fhd = resolution[1][0] * resolution[1][1] / 1e6
mp_4k = resolution[2][0] * resolution[2][1] / 1e6
min_density = 50
write = True


def batch_read(row, idx):
    file_name = row['file_name']
    img = cv2.imread(file_name)
    (h, w, _) = img.shape
    mp = h * w / 1e6

    count = row['count']
    density = count / mp

    # if density < min_density and mp > mp_fhd:
    if mp > mp_4k:
        mat_file_name = file_name.replace('images', 'ground_truth').replace('.jpg', '_ann.mat')
        mat = io.loadmat(mat_file_name)
        gt = np.array(mat['annPoints'])

        file_name = file_name.split('/')[-1].replace('.', '_v2.')
        # if mp > mp_4k:
        height, width = resolution[2][0], resolution[2][1]
        # density_temp = count / (height * width / 1e6)
        # if density_temp < min_density:
        #     height, width = resolution[1][0], resolution[1][1]
        # else:
        #     height, width = resolution[1][0], resolution[1][1]

        resized = resize_image(file_name, img, height, width)
        print(idx, file_name, count, img.shape, mp, density, resized.shape)

        mat_path = mat_file_name.split('/')[-1].replace('.mat', '_v2')
        resize_points(img, resized, gt, mat_path, count)
        print(idx, mat_path, gt.shape, count)

        return True

    return False


def resize_image(file_name, resized, height, width):
    (h, w, _) = resized.shape
    # print(file_name, resized.shape, w / h)

    fx = 1
    if w > width:
        fx = width / w
        h *= fx
        w *= fx
        # print(file_name, resized.shape, w / h)

    fy = 1
    if h > height:
        fy = height / h
        h *= fy
        w *= fy
        # print(file_name, resized.shape, w / h)

    factor = fx * fy
    # print(factor)

    if factor != 1:
        resized = cv2.resize(resized, dsize=(0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    if write:
        cv2.imwrite(out_path + file_name, resized)

    return resized


def resize_points(img, resized, gt, mat_path, count):
    (h0, w0, _) = img.shape
    (h1, w1, _) = resized.shape

    fx = w1 / w0
    fy = h1 / h0

    # print(fx, fy)

    # np.savetxt('gt.csv', gt, delimiter=',')
    # draw_points(img, gt, 'out.jpg')
    resized_gt = gt.copy()
    if count > 0:
        resized_gt[:, 0] *= fx
        resized_gt[:, 1] *= fy

    if write:
        np.save(out_path + mat_path, resized_gt)
    # np.savetxt('resized_gt.csv', resized_gt, delimiter=',')
    # draw_points(resized, resized_gt, 'resized_out.jpg')


def draw_points(img, pts, file_name):
    for pt in pts:
        h = int(pt[0])
        w = int(pt[1])
        img = cv2.circle(img=img, center=(h, w), radius=3, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(file_name, img)


clean_data = []
for idx, row in data.iterrows():
    ret = batch_read(row, idx)
    if ret:
        row['file_name'] = row['file_name'].replace('.', '_v2.')

    row['file_name'] = row['file_name'].split('/')[-1]
    clean_data.append(row)
    print(idx)
    # break

data = pd.DataFrame(clean_data)
data.to_csv('df_labels_clean.csv', index=False)
