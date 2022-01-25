import time

from tensorflow.keras import models

from conv_net_utility import *

# base_path = ''
base_path = 'D:/odrive/grad/traffic_signal/code/aws/'
is_vid = False
num_frame = 10
# multi_v8.2.5_bcep_adam_512_4_ucf_18v1_best
is_sota = False
is_best = True
bench_name = 'ucf_18v1'
version = 'v8.2.5'
lm_loss = 'bcep'
optimizer = 'adam'
INPUT_SIZE = 512
batch_size = 4

postfix = lm_loss
model_path = base_path + 'multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + '_' + \
             str(batch_size) + '_' + bench_name

ext = ''
if is_best:
    ext += '_best'
ext += '.model'

model = models.load_model(model_path + ext, compile=False)

vid_path = 'D:/Dropbox/academic/code/test/hajj_images/hajj_2020_2.mp4'
out_path = 'output/output.avi'

img_path = 'test_img_0092.jpg'
prefix = 'D:/odrive/grad/traffic_signal/dataset/ucf_18/ucf_18_test/'
# prefix = '/content/drive/My Drive/traffic_signal/dataset/ucf_18/ucf_18_test/'
np_path = 'D:/odrive/grad/traffic_signal/dataset/ucf_18/patch_' + str(INPUT_SIZE) + '/'
# np_path = '/content/drive/My Drive/traffic_signal/dataset/ucf_18/patch_' + str(INPUT_SIZE) + '/'
num_dir = 16


def gt_frame(idx, file_name, is_gt):
    start = time.time()

    if not is_gt:
        frame = cv2.imread(file_name) if not is_vid else file_name
        parts, grid = make_patch(frame, INPUT_SIZE, bench_name)
        parts_lm = predict_multi_dm(img_array=parts, model=model, is_multi=False, is_sota=is_sota)
    else:
        grid, parts, _, parts_lm = density_map_patches(file_name=file_name, INPUT_SIZE=INPUT_SIZE, dm_factor=1,
                                                       scale=1, np_path=np_path, num_dir=num_dir, bench_name=bench_name)

    print(parts.shape, parts_lm.shape, grid)

    count, gt_img, lm_map = get_localization_map(parts=parts, pred_parts=parts_lm, grid=grid, is_gt=is_gt,
                                                 is_sota=is_sota)
    end = time.time()

    print(idx, count, gt_img.shape, lm_map.shape, (end - start))

    return gt_img, lm_map


def draw_frame(gt_img, lm_map):
    img = restore_image(gt_img, INPUT_SIZE, bench_name)
    frame = draw_output(img=img, lm_map=lm_map, lm=False, radius=5, color=(0, 0, 255), thickness=-1)
    return frame


def merge_gt_pred_img(gt_img, lm_map1, lm_map2):
    img = restore_image(gt_img, INPUT_SIZE, bench_name)

    frame = draw_output(img=img, lm_map=lm_map1, lm=False, radius=10, color=(0, 255, 0), thickness=2)
    frame = draw_output(img=frame, lm_map=lm_map2, lm=False, radius=5, color=(0, 0, 255), thickness=-1)

    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)

    return frame


if is_vid:
    cap = cv2.VideoCapture(vid_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(np.round(cap.get(cv2.CAP_PROP_FPS)))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(fourcc, fps, (w, h))
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        gt_image, lm_map = gt_frame(i, frame, False)
        frame = draw_frame(gt_image, lm_map)
        frame = np.uint8(frame)
        out.write(frame)

        if cv2.waitKey(fps) & 0xFF == ord('q'):
            break

        i += 1
        if i == num_frame:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

else:
    gt_img, lm_map1 = gt_frame(0, prefix + img_path, True)
    _, lm_map2 = gt_frame(0, prefix + img_path, False)

    frame = merge_gt_pred_img(gt_img, lm_map1, lm_map2)

    cv2.imwrite('output/merged_' + img_path, frame)
