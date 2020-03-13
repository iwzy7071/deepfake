import json
import os
from os.path import join
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing

fake2real = json.load(open('/root/data/wzy/datalocker/fake2real_vedio.json'))
name2path = json.load(open('/root/data/wzy/datalocker/video_name_to_path.json'))
name2json = json.load(open('/root/data/wzy/datalocker/name2json.json'))

image_root_path1 = '/root/disk4/deepfake/dataset/DFDC_images'
image_root_path2 = '/root/disk3/DFDC_images'
save_path = '/root/disk2/multitask_mask_2020_02_27'


def get_crop_box(box, frame, factors=0.5):
    left, top, right, bottom = box
    height, width = frame.shape[:2]
    left, right, bottom, top = max(left, 0), min(width, right), min(height, bottom), max(0, top)
    face_width = right - left
    face_height = bottom - top

    left = max(left - face_width * factors, 0)
    right = min(width, right + face_width * factors)
    bottom = min(height, bottom + face_height * factors)
    top = max(0, top - face_height * factors)

    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    return left, top, right, bottom


def get_save_path(path):
    save_path = path.strip('.mp4')
    save_path = '/'.join(save_path.split('/')[-2:])
    if os.path.exists(join(image_root_path1, save_path)):
        save_path = join(image_root_path1, save_path)
    elif os.path.exists(join(image_root_path2, save_path)):
        save_path = join(image_root_path2, save_path)
    else:
        return None
    return save_path


def crop_video2_image(fake2real):
    for fake_name, real_name in tqdm(fake2real):
        if fake_name not in name2path or real_name not in name2path:
            continue
        # GET MASK IMAGE
        fake_path, real_path = name2path[fake_name], name2path[real_name]
        fake_path, real_path = get_save_path(fake_path), get_save_path(real_path)
        if fake_path is None or real_path is None:
            continue
        fake_images, real_images = os.listdir(fake_path), os.listdir(real_path)
        fake_images, real_images = sorted(fake_images), sorted(real_images)

        nums = np.linspace(start=0, stop=len(fake_images) - 1, num=8, dtype=int)
        fake_images = [fake_images[num] for num in nums]
        real_images = [real_images[num] for num in nums]
        fake_images = [join(fake_path, path) for path in fake_images]
        real_images = [join(real_path, path) for path in real_images]

        real_images = [cv2.imread(real_image) for real_image in real_images]
        fake_images = [cv2.imread(fake_image) for fake_image in fake_images]
        # GET FACE BOX
        real_json_path = name2json[real_name]

        # GET SAVE PATH
        current_save_mask_dir = join(save_path, 'mask', fake_name)
        current_save_real_dir = join(save_path, 'real', real_name)
        current_save_fake_dir = join(save_path, 'fake', fake_name)
        os.makedirs(current_save_fake_dir, exist_ok=True)
        os.makedirs(current_save_real_dir, exist_ok=True)
        os.makedirs(current_save_mask_dir, exist_ok=True)

        for box_index, boxes in enumerate(json.load(open(real_json_path))):
            boxes = [boxes[num] for num in nums]
            for face_index, (real_image, fake_image, box) in enumerate(zip(real_images, fake_images, boxes)):
                left, top, right, bottom = get_crop_box(box, real_image)
                real_face = real_image[top:bottom, left:right, :]
                fake_face = fake_image[top:bottom, left:right, :]
                mask = cv2.absdiff(fake_face, real_face)
                if mask.sum() == 0:
                    print("AAAAAAAA")
                    exit()

fake2real = [[key, fake2real[key]] for key in fake2real.keys()]
result = []
for index in range(0, 10):
    result.append(fake2real[index * 10000:(index + 1) * 10000])

processes = []
for res in result:
    process = multiprocessing.Process(target=crop_video2_image, args=(res,))
    processes.append(process)
    process.start()

[process.join() for process in processes]
