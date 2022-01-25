import time
from ctypes import *

import cv2
import numpy as np
import pandas as pd


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

folder_name = 'benchmark'

configPath = ['data/class_csv_dws.cfg', 'data/class_csv_dws2.cfg']
weightPath = ['data/class_csv_dws_final.weights', 'data/class_csv_dws2_final.weights']

num_of_classes = 15

IMG_SIZE = 128
top_k = 1


def array_to_image(arr):
    arr = arr.transpose(2, 0, 1)
    (c, h, w) = arr.shape
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im


def image_patch(image_name):
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (h, w, c) = img.shape

    xgrid = np.floor(w / IMG_SIZE)
    ygrid = np.floor(h / IMG_SIZE)

    xstep = int(np.ceil(w / xgrid))
    ystep = int(np.ceil(h / ygrid))

    parts = []
    for y in range(0, h, ystep):
        for x in range(0, w, xstep):
            xmin = x
            ymin = y
            xmax = xmin + xstep
            ymax = ymin + ystep

            cp_im = img[ymin:ymax, xmin:xmax]
            new_array = cv2.resize(cp_im, (IMG_SIZE, IMG_SIZE))
            parts.append(new_array)

    return parts


def predict(parts, net):
    predictions = []
    for part in parts:
        im = array_to_image(part)
        out = predict_image(net, im)[0:num_of_classes]
        predictions.append(out)

    return predictions


def class_to_count(prediction):
    avg_val = [0, 2.5, 6.5, 12.5, 20, 29, 40.5, 54.5, 71.5, 93, 119.5, 152, 192, 241.5, 322]
    classes = np.argmax(prediction, axis=-1)
    total = 0
    for c in classes:
        total += avg_val[c]

    return total


shallow_net = load_net_custom(configPath[0].encode("ascii"), weightPath[0].encode("ascii"), 0, 1)  # batch size = 1
deep_net = load_net_custom(configPath[1].encode("ascii"), weightPath[1].encode("ascii"), 0, 1)  # batch size = 1

prefix = folder_name + '/'
data = pd.read_csv(prefix + folder_name + '.csv')
data['file_name'] = data['file_name'].apply(lambda name: prefix + name)

cls_out = []
for idx, row in data.iterrows():
    file_name = row['file_name']
    start = time.time()
    parts = image_patch(file_name)
    com = time.time()

    predictions = predict(parts, shallow_net)
    count_sh = class_to_count(predictions)
    sh = time.time()

    predictions = predict(parts, deep_net)
    count_dp = class_to_count(predictions)
    dp = time.time()

    out = file_name.split('/')[-1] + ',' + str(count_sh) + ',' + str(count_dp) + ',' \
          + str(sh - start) + ',' + str(dp - sh + com - start)
    cls_out.append(out)
    print(idx, out)

with open('darknet_out.csv', 'w') as csv_file:
    for out in cls_out:
        csv_file.write(out + '\n')
