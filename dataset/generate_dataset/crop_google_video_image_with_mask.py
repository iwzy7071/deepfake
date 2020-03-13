from os.path import join
import os
import cv2
import threading
from tqdm import tqdm
import numpy as np


def cropimage2face(image_path, mask_image_path):
    save_path = '/root/disk2/google_fake2mask_box'
    dir_path = image_path.split('/')[-2]
    image_name = image_path.split('/')[-1]

    factors = 0.5
    mask_img = cv2.imread(mask_image_path)
    ori_img = cv2.imread(image_path)
    Height, Width = ori_img.shape[:2]
    bb = np.nonzero(mask_img)
    if len(bb[0]) == 0:
        return
    cc = np.transpose(bb)
    top = np.min(cc[:, 0])
    bottom = np.max(cc[:, 0])
    left = np.min(cc[:, 1])
    right = np.max(cc[:, 1])
    face_width = right - left
    face_height = bottom - top
    left = max(0, left - face_width * factors)
    right = min(Width, right + face_width * factors)
    top = max(0, top - face_height * factors)
    bottom = min(Height, bottom + face_height * factors)

    face = ori_img[int(top):int(bottom), int(left):int(right), :]
    mask = mask_img[int(top):int(bottom), int(left):int(right), :]
    save_mask_path = join(save_path, 'mask', dir_path)
    save_face_path = join(save_path, 'face', dir_path)
    os.makedirs(save_face_path, exist_ok=True)
    os.makedirs(save_mask_path, exist_ok=True)
    cv2.imwrite(join(save_mask_path, image_name), mask)
    cv2.imwrite(join(save_face_path, image_name), face)


root_path = '/root/disk2/GoogleDetection/DeepFakeDetection'
mask_path = '/root/disk2/GoogleDetection/DeepFakeDetection/masks/images'
comps = ['c23', 'c40']
count = 0
for comp in comps:
    part_path = join(root_path, comp, 'images')
    for video_name in tqdm(os.listdir(part_path)):
        mask_video_path = join(mask_path, video_name)
        video_path = join(part_path, video_name)
        for image_name in os.listdir(video_path):
            image_path = join(video_path, image_name)
            mask_image_path = join(mask_video_path, image_name)
            if not os.path.exists(mask_image_path) or not os.path.exists(image_path):
                continue
            cropimage2face(image_path, mask_image_path)
