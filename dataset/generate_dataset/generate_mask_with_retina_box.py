from os.path import join
from tqdm import tqdm
import re
import json
import cv2
import numpy as np
import os

root_path = '/root/disk2/new_tracking_image_large_factor'
dfdc_path_a = '/root/disk3/DFDC_images'
dfdc_path_b = '/root/disk4/deepfake/dataset/DFDC_images'

real_result = []
fake_result = []
for mask_name in mask_names:
    landmark_path = join(root_path, mask_name)

    landmark_name_end = mask_name.index('.')
    landmark_name_start = re.search('valid_face', mask_name).end()
    landmark = mask_name[landmark_name_start:landmark_name_end].split('-')
    landmark_start, landmark_end = int(landmark[0]), int(landmark[1])

    labels = {}
    for index in range(landmark_start, landmark_end):
        dfdc_path = 'dfdc_train_part_{}'.format(index)
        dfdc_json_path = join('/root/disk4/deepfake/dataset/DFDC_videos/DFDC_videos', dfdc_path, 'metadata.json')
        datajson = json.load(open(dfdc_json_path, 'r'))
        labels.update(datajson)

    for line in tqdm(open(landmark_path, 'r').readlines()):
        line = line.strip('\n').split(',')
        image_path = line[0]
        vedio_name = image_path.split('/')[5] + '.mp4'
        if vedio_name not in labels:
            continue

        label = labels[vedio_name]['label']
        if label != 'FAKE':
            real_result.append(image_path)
            continue

        image_left, image_right, image_top, image_bottom, mask_left, mask_right, mask_top, mask_bottom = line[1:]

        image_left, image_right, image_top, image_bottom = \
            int(image_left), int(image_right), int(image_top), int(image_bottom)
        image_width, image_height = image_right - image_left, image_bottom - image_top
        mask_img = np.zeros((image_bottom, image_right))
        mask_left, mask_right, mask_top, mask_bottom = int(mask_left), int(mask_right), int(mask_top), int(mask_bottom)
        mask_img[mask_top:mask_bottom, mask_left:mask_right] = 255
        mask_img = mask_img[image_top:image_bottom, image_left:image_right]

        mask_dir_name = image_path.split('/')[4:]
        mask_dir_name = '/'.join(mask_dir_name)
        mask_save_path = join('/root/data/dfdc_mask', mask_dir_name)
        mask_save_dir = '/'.join(mask_save_path.split('/')[:-1])

        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir, exist_ok=True)

        cv2.imwrite(mask_save_path, mask_img)
        fake_result.append([image_path, mask_save_path])
