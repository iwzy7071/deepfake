import cv2
import json
from os.path import join
import os
from tqdm import tqdm
import threading
import numpy as np

landmasks_root = '/root/disk4/deepfake/dataset/face_info/new_landmark'

factors = 0.3


def load_video_to_face(video_path, part_json):
    global factors
    if '/root/disk4/deepfake' in video_path:
        save_path = video_path.replace("/root/disk4/deepfake/dataset/DFDC_images",
                                       "/root/disk2/new_tracking_image_small_factor_mtcnn")
    else:
        save_path = video_path.replace("/root/disk3/DFDC_images",
                                       "/root/disk2/new_tracking_image_small_factor_mtcnn")

    os.makedirs(save_path, exist_ok=True)

    frame_names = sorted(os.listdir(video_path))
    frames_nums = np.linspace(0, len(frame_names) - 1, 8, dtype=int).tolist()
    frame_names = [frame_names[num] for num in frames_nums]

    for frame_name in frame_names:
        if frame_name not in part_json:
            continue

        box = part_json[frame_name]['box']
        score = part_json[frame_name]['score']
        if len(score) == 0:
            continue
        max_score = max(score)
        if max_score < 0.95:
            continue

        current_box = box[score.index(max_score)]
        frame_path = join(video_path, frame_name)
        frame = cv2.imread(frame_path)

        left, top, right, bottom = int(current_box[0]), int(current_box[1]), int(current_box[2]), int(current_box[3])
        height, width = frame.shape[:2]

        left = max(left, 0)
        right = min(width, right)
        bottom = min(height, bottom)
        top = max(0, top)
        face_width = right - left
        face_height = bottom - top

        left = int(max(left - face_width * factors, 0))
        right = int(min(width, right + face_width * factors))
        bottom = int(min(height, bottom + face_height * factors))
        top = int(max(0, top - face_height * factors))

        face = frame[top:bottom, left:right, :]
        cv2.imwrite(join(save_path, '{}.png').format(frame_name), face)


def get_landmark_json(index):
    data = {}
    landmark_name = 'landmarks.%02d.txt' % index
    landmark_path = join(landmasks_root, landmark_name)
    for line in open(landmark_path).readlines():
        img_path, attribute = line.strip('\n').split('\t')
        attribute = eval(attribute)
        img_path = img_path.split('/')[-1]
        data[img_path] = {'box': attribute['bbox'], 'score': attribute['score']}
    return data


for index in tqdm(range(0, 50)):
    part_json = get_landmark_json(index)
    part_name = 'dfdc_train_part_{}'.format(index)
    part_path = join('/root/disk3/DFDC_images', part_name) if os.path.exists(
        join('/root/disk3/DFDC_images', part_name)) else join('/root/disk4/deepfake/dataset/DFDC_images', part_name)
    video_name = os.listdir(part_path)
    video_paths = [join(part_path, name) for name in video_name]
    threads = []
    for video_path in video_paths:
        thread = threading.Thread(target=load_video_to_face, args=(video_path, part_json))
        threads.append(thread)
        thread.start()
        if len(threads) > 500:
            [thread.join() for thread in threads]
            threads = []

    [thread.join() for thread in threads]
