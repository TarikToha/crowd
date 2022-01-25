import argparse
import os
import time

import mysql.connector as con
import pandas as pd
from tensorflow.keras import models

from conv_net_utility import *

debug = False

resolution = [[720, 1280], [1080, 1920], [2160, 3840]]
mp_fhd = resolution[1][0] * resolution[1][1] / 1e6
mp_4k = resolution[2][0] * resolution[2][1] / 1e6


def load_model(model_path):
    model = models.load_model(model_path, compile=False)
    return model


def load_image(file_name):
    image = cv2.imread(file_name)
    (h, w, _) = image.shape
    mp = h * w / 1e6

    resized = None
    if mp > mp_fhd:
        height, width = resolution[1][0], resolution[1][1]
        resized = resize_image(image, height, width)

    return image, resized


def resize_image(resized, height, width):
    (h, w, _) = resized.shape

    fx = 1
    if w > width:
        fx = width / w
        h *= fx
        w *= fx

    fy = 1
    if h > height:
        fy = height / h
        h *= fy
        w *= fy

    factor = fx * fy
    if factor != 1:
        resized = cv2.resize(resized, dsize=(0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    return resized


def save_image_pts(file_name, image, lm_pts):
    file_name, ext = os.path.splitext(file_name)
    out_path = file_name + '_out' + ext

    cv2.imwrite(out_path, image)

    lm_pts = pd.DataFrame(lm_pts, columns=['x_coord', 'y_cord'])
    lm_pts.to_csv(file_name + '_pts.csv', index=False)


def run_model(image, model):
    parts, grid = make_patch(image, 512, 'ucf_18v1')

    parts_lm = predict_multi_dm(img_array=parts, model=model, is_multi=False, is_sota=False)

    count, _, dot_map = get_localization_map(parts=parts, pred_parts=parts_lm, grid=grid, is_gt=False, is_sota=False)

    return count, dot_map


def draw_image(image, resized, dot_map, count, radius):
    image, lm_pts = draw_output(img=image, resized=resized, count=count, lm_map=dot_map, lm=False, radius=radius,
                                color=(0, 0, 255), thickness=-1)
    return image, lm_pts


def update_database(count, img_id, pred_time):
    mydb = con.connect(host="localhost", user="root", password="root", database="deeps")

    mycursor = mydb.cursor()

    sql = "UPDATE images SET count = %s, time = %s WHERE id = %s"
    val = (count, pred_time, img_id)

    mycursor.execute(sql, val)
    mydb.commit()


def driver(img_path, model_path, img_id):
    image, resized = load_image(img_path)
    model = load_model(model_path)

    start = time.time()
    if resized is not None:
        count, dot_map = run_model(resized, model)
    else:
        count, dot_map = run_model(image, model)
    end = time.time()

    if resized is not None:
        image, lm_pts = draw_image(image, resized.shape, dot_map, count, 10)
    else:
        image, lm_pts = draw_image(image, None, dot_map, count, 5)

    save_image_pts(img_path, image, lm_pts)

    count = int(np.array(count))
    if not debug:
        update_database(count, img_id, end - start)
    else:
        print(count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='localization_web.py')

    if not debug:
        parser.add_argument('img_path', type=str, help='input image path')
        parser.add_argument('model_path', type=str, help='input model path')
        parser.add_argument('img_id', type=str, help='input image id')
        arg = parser.parse_args()
        img = arg.img_path
        ml = arg.model_path
        id = arg.img_id
    else:
        img = 'hajj_images/rob-curran-sUXXO3xPBYo-unsplash.jpg'
        ml = 'best_models/multi_v8.2.5_bcep_adam_512_4_ucf_18v1_best.model'
        id = 10

    driver(img, ml, id)
