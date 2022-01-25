from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import generate_binary_structure, maximum_filter, binary_erosion
from scipy.signal import windows
from scipy.spatial import KDTree
from skimage.metrics import structural_similarity
from tensorflow import equal, cast
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.keras.models import Model

from db_mean_std import *

debug = False


def predict_dm(img_array, model):
    prediction = model.predict(img_array)

    prediction = np.squeeze(prediction, axis=-1)

    return prediction


def predict_layers(img_array, model, layer_id):
    layer = model.layers[layer_id]
    print(layer.name)
    model = Model(inputs=model.inputs, outputs=layer.output)

    prediction = model.predict(img_array)

    return prediction


def predict_multi_dm(img_array, model, is_multi, is_sota):
    prediction = model.predict(x=img_array, batch_size=4)

    if is_multi:
        prediction[0] = np.squeeze(prediction[0], axis=-1)
        y_pred = prediction[1]
    else:
        y_pred = prediction

    if not is_sota:
        y_pred = convert_to_tensor(y_pred)
        y_pred = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y_pred)
        y_pred = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y_pred)
        y_pred = y_pred.numpy()
    else:
        y_pred = convert_to_tensor(y_pred)
        y_pred = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y_pred)
        y_max = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y_pred)
        y_max = equal(y_max, y_pred)
        y_max = cast(y_max, dtype='float32')
        y_pred *= y_max
        # y_pred = round(y_pred)
        y_pred = y_pred.numpy()

    if is_multi:
        prediction[1] = np.squeeze(y_pred, axis=-1)
    else:
        prediction = np.squeeze(y_pred, axis=-1)

    return prediction


def class_to_count(prediction, bench_name):
    # avg_val = [0, 2.5, 6.5, 12.5, 20, 29, 40.5, 54.5, 71.5, 93, 119.5, 152, 192, 241.5, 322]
    # avg_val = [0, 3, 8.5, 16, 25, 36, 49, 64, 81, 101, 125.5, 155.5, 194.5, 243, 322]  # shanghai_A
    # avg_val = [0, 2, 4, 6.5, 10, 14, 18.5, 24, 31, 40, 52, 67.5, 89, 117.5, 154]  # shanghai_B
    if bench_name == 'ucf_13_0':
        avg_val = [0, 5.5, 16.5, 30, 43.5, 56.5, 71.5, 91, 116, 146.5, 179.5, 216, 256.5, 326]
    elif bench_name == 'ucf_13_1':
        avg_val = [0, 4, 11, 20, 31, 43, 55, 69, 87, 110.5, 139.5, 173.5, 214, 280.5]
    elif bench_name == 'ucf_18_dense':
        avg_val = [0, 4, 12, 24, 40, 60, 84.5, 114.5, 151.5, 197, 252, 316.5, 393.5, 496, 676.5, 942, 1268, 1724.5]
    else:
        raise Exception("average values are not given")

    classes = np.argmax(prediction, axis=-1)
    # print(classes)
    total = 0
    for c in classes:
        total += avg_val[c]

    return total


def image_patches(file_name, PATCH_SIZE, INPUT_SIZE):
    img = cv2.imread(file_name)
    (h, w, c) = img.shape
    # fig = plt.figure()

    xgrid = max(1, np.floor(w / PATCH_SIZE))
    ygrid = max(1, np.floor(h / PATCH_SIZE))

    xstep = int(np.ceil(w / xgrid))
    ystep = int(np.ceil(h / ygrid))

    # count = 0
    parts = []
    for y in range(0, h, ystep):
        for x in range(0, w, xstep):
            xmin = x
            ymin = y
            xmax = xmin + xstep
            ymax = ymin + ystep
            # print(xmin, ymin, xmax, ymax)

            cp_im = img[ymin:ymax, xmin:xmax]
            new_array = cv2.resize(cp_im, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            parts.append(new_array)

            # count = count + 1
            # fig.add_subplot(grid, grid, count)
            # plt.imshow(cp_im)
            # plt.axis('off')
            # print(count)
    return parts


def density_map_patches(file_name, INPUT_SIZE, dm_factor, scale, np_path, num_dir, bench_name):
    img = cv2.imread(file_name)
    (h, w, c) = img.shape

    bottom = int(np.ceil(h / INPUT_SIZE)) * INPUT_SIZE - h
    right = int(np.ceil(w / INPUT_SIZE)) * INPUT_SIZE - w

    img = cv2.copyMakeBorder(img, top=0, bottom=bottom, left=0,
                             right=right, borderType=cv2.BORDER_CONSTANT)
    (h, w, c) = img.shape

    grid = (h // INPUT_SIZE, w // INPUT_SIZE)

    # depends on dataset
    mean, std = get_mean_std(INPUT_SIZE=INPUT_SIZE, bench_name=bench_name)

    if mean is None or std is None:
        raise Exception("pixel mean and std values are not given")

    if 'shanghai' in bench_name:
        base_name = file_name.split('/')[-1].replace('IMG', 'GT_IMG').replace('.jpg', '_' + str(INPUT_SIZE) + '.npy')
    elif 'nwpu' in bench_name:
        base_name = file_name.split('/')[-1].replace('.jpg', '_' + str(INPUT_SIZE) + '.npy')
    else:
        base_name = file_name.split('/')[-1].replace('.jpg', '_ann_' + str(INPUT_SIZE) + '.npy')

    OUTPUT_SIZE = INPUT_SIZE // dm_factor

    count = 0
    parts = []
    parts_dm = []
    parts_lm = []
    for y in range(0, h, INPUT_SIZE):
        for x in range(0, w, INPUT_SIZE):
            xmin = x
            ymin = y
            xmax = xmin + INPUT_SIZE
            ymax = ymin + INPUT_SIZE
            # print(xmin, ymin, xmax, ymax)

            cp_im = img[ymin:ymax, xmin:xmax]
            cp_im = cp_im / 255
            for c in range(3):
                cp_im[:, :, c] = (cp_im[:, :, c] - mean[c]) / std[c]

            if scale > 1:
                cp_im = cv2.resize(cp_im, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            parts.append(cp_im)

            patch_id = str(count % num_dir)
            count = count + 1
            np_file_name = np_path + 'patch_dm' + patch_id + '/' + base_name. \
                replace('.', '_' + str(count) + '.')

            if '_v2_ann_' in np_file_name:
                np_file_name = np_file_name.replace('_v2_ann_', '_ann_v2_')

            # density_map = np.load(np_file_name, allow_pickle=True)
            # if dm_factor > 1:
            #     density_map = cv2.resize(density_map, (OUTPUT_SIZE, OUTPUT_SIZE),
            #                              interpolation=cv2.INTER_AREA) * dm_factor * dm_factor
            #
            # parts_dm.append(density_map)

            lm_file_name = np_file_name.replace('patch_dm', 'patch_lm')
            point_map = np.load(lm_file_name, allow_pickle=True)
            parts_lm.append(point_map)

    parts = np.array(parts)
    parts_dm = np.array(parts_dm)
    parts_lm = np.array(parts_lm)
    return grid, parts, parts_dm, parts_lm


def make_patch(img, INPUT_SIZE, bench_name):
    (h, w, c) = img.shape

    bottom = int(np.ceil(h / INPUT_SIZE)) * INPUT_SIZE - h
    right = int(np.ceil(w / INPUT_SIZE)) * INPUT_SIZE - w

    img = cv2.copyMakeBorder(img, top=0, bottom=bottom, left=0,
                             right=right, borderType=cv2.BORDER_CONSTANT)
    (h, w, c) = img.shape

    grid = (h // INPUT_SIZE, w // INPUT_SIZE)

    # depends on dataset
    mean, std = get_mean_std(INPUT_SIZE=INPUT_SIZE, bench_name=bench_name)

    if mean is None or std is None:
        raise Exception("pixel mean and std values are not given")

    parts = []
    for y in range(0, h, INPUT_SIZE):
        for x in range(0, w, INPUT_SIZE):
            xmin = x
            ymin = y
            xmax = xmin + INPUT_SIZE
            ymax = ymin + INPUT_SIZE
            # print(xmin, ymin, xmax, ymax)

            cp_im = img[ymin:ymax, xmin:xmax]
            cp_im = cp_im / 255
            for c in range(3):
                cp_im[:, :, c] = (cp_im[:, :, c] - mean[c]) / std[c]

            parts.append(cp_im)

    parts = np.array(parts)
    return parts, grid


def psnr_custom(y_true, y_pred):
    mse = np.mean(np.square(y_true - y_pred + 1e-8))
    psnr = 20 * np.log10(1 / np.sqrt(mse))
    return psnr


def batch_psnr_ssim(y_true, y_pred, res, i):
    psnr = psnr_custom(y_true, y_pred)
    ssim = structural_similarity(y_true, y_pred)
    res[i] = [psnr, ssim]


def compute_psnr_ssim(y_true, y_pred):
    num_part = y_true.shape[0]
    res = np.zeros((num_part, 2))
    thread_list = []
    for i in range(num_part):
        t = Thread(target=batch_psnr_ssim, args=(y_true[i], y_pred[i], res, i))
        t.start()
        thread_list.append(t)

    for t in thread_list:
        t.join()

    res = np.mean(res, axis=0)
    return res


def get_peak_map_points(img, is_gt, is_sota):
    if is_gt:
        neighborhood = generate_binary_structure(2, 1)
        peak_map = binary_erosion(img, structure=neighborhood)
    elif is_sota:
        img = img.copy()
        img[img < 0.5] = 0
        peak_map = img
    else:
        img = img.copy()
        img[img < 0.5] = 0

        neighborhood = generate_binary_structure(2, 2)
        local_max = maximum_filter(img, footprint=neighborhood) == img
        background = (img == 0)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        peak_map = local_max ^ eroded_background
        peak_map = binary_erosion(peak_map, structure=neighborhood)

    peak_points = np.nonzero(peak_map)
    peak_points = list(zip(peak_points[1], peak_points[0]))
    return peak_map, peak_points


def get_true_positives(pred_idx, distances, gt_count, pred_count, k):
    neighbor = []
    assigned = np.zeros(gt_count)
    taken = np.zeros(pred_count)
    for it in range(k):
        pts_dist = np.array([np.arange(gt_count), pred_idx[:, it], distances[:, it]]).T
        pts_dist = sorted(pts_dist, key=lambda element: element[2])

        for pt in pts_dist:
            if pt[2] == np.inf:
                continue
            src = int(pt[0])
            dst = int(pt[1])
            if assigned[src] == 0 and taken[dst] == 0:
                neighbor.append(pt)
                assigned[src] = 1
                taken[dst] = 1

    tp = len(neighbor)
    loc_error = 0
    if tp > 0:
        neighbor = np.array(neighbor)
        loc_error = neighbor.sum(axis=0)[2]

    return tp, loc_error


def get_kernel(kernlen=49, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = windows.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def gaussian_function(INPUT_SIZE, gt_point, pred_point, sigma):
    point_map = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    point_map[gt_point[1], gt_point[0]] = 1

    kernel = get_kernel(std=sigma)
    point_map = cv2.filter2D(src=point_map, ddepth=-1, kernel=kernel)
    point_map /= point_map.max()
    val = point_map[pred_point[1], pred_point[0]]
    return val, point_map


def get_true_positives_sota(pred_prob, true_points, gt_idx, threshold, conf_score, INPUT_SIZE, k):
    gt_count = len(true_points)
    assigned = np.zeros(gt_count)
    for i, p in enumerate(pred_prob):
        for it in range(k):
            gt = gt_idx[i, it]

            if gt >= gt_count:
                break

            if assigned[gt] == 1:
                continue

            gt_point = true_points[gt]
            pred_point = int(p[0]), int(p[1])
            val, _ = gaussian_function(INPUT_SIZE=INPUT_SIZE, gt_point=gt_point, pred_point=pred_point, sigma=threshold)

            if val >= conf_score:
                assigned[gt] = 1
                break

    tp = np.count_nonzero(assigned)
    return tp


def batch_localize(y_true, y_pred, tp_gt_pred, i, threshold, k, conf_score, is_sota):
    y_true2, true_points = get_peak_map_points(y_true, True, False)
    gt_count = len(true_points)

    y_pred2, pred_points = get_peak_map_points(y_pred, False, is_sota)
    pred_count = len(pred_points)

    if debug:
        print(gt_count, pred_count)
        if gt_count != 0:
            compare_chart(y_true2, y_pred2)

    tp, loc_error = 0, 0
    if gt_count > 0 and pred_count > 0:

        if not is_sota:
            tree = KDTree(pred_points, leafsize=64)
            distances, pred_idx = tree.query(true_points, k=k, distance_upper_bound=threshold)
            tp, loc_error = get_true_positives(pred_idx, distances, gt_count, pred_count, k)

        else:
            pred_prob = []
            for p in pred_points:
                val = y_pred2[p[1], p[0]]
                pred_prob.append([p[0], p[1], val])

            pred_prob = np.array(sorted(pred_prob, key=lambda element: element[2], reverse=True))
            pred_points = pred_prob[:, 0:2]

            tree = KDTree(true_points, leafsize=64)
            _, gt_idx = tree.query(pred_points, k=k)
            tp = get_true_positives_sota(pred_prob, true_points, gt_idx, threshold, conf_score, y_pred2.shape[0], k)

    tp_gt_pred[i] = [tp, gt_count, pred_count, loc_error]


def compute_precision_recall(y_true, y_pred, threshold, conf_score, is_sota):
    k = 4
    num_part = y_true.shape[0]
    tp_gt_pred = np.zeros((num_part, 4))
    thread_list = []
    for i in range(num_part):
        if debug:
            batch_localize(y_true[i], y_pred[i], tp_gt_pred, i, threshold, k, conf_score, is_sota)
        else:
            t = Thread(target=batch_localize,
                       args=(y_true[i], y_pred[i], tp_gt_pred, i, threshold, k, conf_score, is_sota))
            t.start()
            thread_list.append(t)

    for t in thread_list:
        t.join()

    tp_gt_pred = np.sum(tp_gt_pred, axis=0)

    tp = tp_gt_pred[0]
    gt_count = tp_gt_pred[1]
    pred_count = tp_gt_pred[2]

    fp = pred_count - tp
    fn = gt_count - tp
    loc_error = tp_gt_pred[3] + max(fp, fn) * 16

    precision = tp / (pred_count + 1e-8)
    recall = tp / (gt_count + 1e-8)
    mle = loc_error / gt_count

    pr = [precision, recall, mle]
    return pred_count, pr


def batch_peak_map_points(part, lm_maps, count_arr, i, is_gt, is_sota):
    lm_map, lm_points = get_peak_map_points(part, is_gt, is_sota)
    count_arr[i] = len(lm_points)
    lm_maps[i] = lm_map


def get_localization_map(parts, pred_parts, grid, is_gt, is_sota):
    num_part = grid[0] * grid[1]
    lm_maps = np.zeros(pred_parts.shape)
    count_arr = np.zeros(num_part)

    thread_list = []
    for i in range(num_part):
        # batch_peak_map_points(pred_parts[i], lm_maps, count_arr, i, conf_score)
        t = Thread(target=batch_peak_map_points, args=(pred_parts[i], lm_maps, count_arr, i, is_gt, is_sota))
        t.start()
        thread_list.append(t)

    for t in thread_list:
        t.join()

    h, w = grid[0], grid[1]

    gt_col = []
    lm_col = []
    idx = 0
    for i in range(h):
        gt_row = []
        lm_row = []
        for j in range(0, w):
            gt_row.append(parts[idx])
            lm_row.append(lm_maps[idx])
            idx = idx + 1

        gt_row = np.hstack(gt_row)
        gt_col.append(gt_row)

        lm_row = np.hstack(lm_row)
        lm_col.append(lm_row)

    gt_img = np.vstack(gt_col)
    lm_map = np.vstack(lm_col)

    count = count_arr.sum()
    return count, gt_img, lm_map


def scale_points(img, resized, lm_pts, count):
    (h0, w0, _) = img
    (h1, w1, _) = resized

    fx = w0 / w1
    fy = h0 / h1

    if count > 0:
        lm_pts[:, 0] = lm_pts[:, 0] * fx
        lm_pts[:, 1] = lm_pts[:, 1] * fy

    return lm_pts


def draw_output(img, resized, count, lm_map, lm, radius, color, thickness):
    if lm:
        plt.imshow(lm_map)
        plt.show()

        return lm_map
    else:
        lm_pts = np.nonzero(lm_map)
        lm_pts = np.array(list(zip(lm_pts[1], lm_pts[0])))
        if resized is not None:
            lm_pts = scale_points(img.shape, resized, lm_pts, count)

        for pt in lm_pts:
            img = cv2.circle(img=img, center=(pt[0], pt[1]), radius=radius, color=color, thickness=thickness)

        return img, lm_pts


def restore_image(cp_im, INPUT_SIZE, bench_name):
    mean, std = get_mean_std(INPUT_SIZE=INPUT_SIZE, bench_name=bench_name)
    for c in range(3):
        cp_im[:, :, c] = cp_im[:, :, c] * std[c] + mean[c]

    cp_im = cp_im * 255

    return cp_im


def visualize_map(y_img, y_lm, y_pred, INPUT_SIZE, bench_name, layer_id):
    y_img = restore_image(y_img, INPUT_SIZE, bench_name)
    y_img = cv2.cvtColor(src=np.float32(y_img), code=cv2.COLOR_BGR2RGB)
    y_img = y_img.astype(np.uint8)

    # fig = plt.figure()

    # fig.add_subplot(1, 2, 1)
    # plt.imshow(y_img)
    # plt.xlabel('y_img')
    # plt.imsave('y_img.png', y_img)

    # fig.add_subplot(1, 2, 1)
    # y_pred = y_pred[:, :, 0]
    # plt.imshow(y_pred)
    # plt.imsave('y_pred_' + str(layer_id) + '.png', y_pred)
    #
    # fig.add_subplot(1, 2, 2)
    y_pred = localized_map(y_pred)
    y_pred = y_pred[:, :, 0]
    # plt.imshow(y_pred)
    plt.imsave('y_pred_' + str(layer_id) + '_loc.png', y_pred)

    # fig.add_subplot(1, 2, 2)
    # plt.imshow(y_lm)
    # plt.imsave('y_lm.png', y_lm)

    # plt.show()


def localized_map(y_pred):
    y_pred = np.expand_dims(y_pred, axis=0)

    y_pred = convert_to_tensor(y_pred)
    y_pred = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y_pred)
    y_max = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y_pred)
    y_max = equal(y_max, y_pred)
    y_max = cast(y_max, dtype='float32')
    y_pred *= y_max
    y_pred = round(y_pred)
    y_pred = y_pred.numpy()

    y_pred = np.squeeze(y_pred, axis=0)
    return y_pred


def compare_chart(y_true, y_pred):
    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.imshow(y_true)
    plt.xlabel('y_true')

    fig.add_subplot(1, 2, 2)
    plt.imshow(y_pred)
    plt.xlabel('y_pred')

    # np.save('y_true', y_true)
    # np.save('y_pred', y_pred)
    plt.show()

    print(y_true.max(), y_pred.max())
