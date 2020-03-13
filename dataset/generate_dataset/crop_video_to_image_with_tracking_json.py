import cv2
import json
from os.path import join
import os
from tqdm import tqdm
import threading
import numpy as np

landmasks_root = '/root/disk2/tracking_landmarks/DFDC_face_tracking/'
frames_root = '/root/disk3/DFDC_images'
video_open_error = []
factors = 0.5


def load_video_to_face(video_path, json_path):
    boxes = json.load(open(json_path))
    save_path = json_path.replace("/root/disk2/tracking_landmarks/DFDC_face_tracking",
                                  "/root/disk2//root/disk2/faces_with_no_widen")
    save_path = save_path[:save_path.index('.')]
    os.makedirs(save_path, exist_ok=True)
    frame_names = os.listdir(video_path)
    frames_num = np.linspace(0, len(frame_names) - 1, 8, dtype=int).tolist()
    frame_names = [frame_names[num] for num in frames_num]
    for frame_name in frame_names:
        frame_path = join(video_path, frame_name)
        frame_number = frame_name.split('_')[-1]
        frame_number = int(frame_number[:frame_number.index('.')])

        frame = cv2.imread(frame_path)

        for bbox in boxes:
            left, top, right, bottom = int(bbox[frame_number - 1][0]), int(bbox[frame_number - 1][1]), \
                                       int(bbox[frame_number - 1][2]), int(bbox[frame_number - 1][3])
            height, width = frame.shape[:2]

            left = max(left, 0)
            right = min(width, right)
            bottom = min(height, bottom)
            top = max(0, top)

            face = frame[top:bottom, left:right, :]
            cv2.imwrite(join(save_path, '_{}_{}_.png').format(frame_number - 1, boxes.index(bbox)), face)


threads = []
for part_number in tqdm(range(0, 50)):
    part_name = 'dfdc_train_part_{}'.format(part_number)
    part_landmark_path = join(landmasks_root, part_name)
    part_frame_path = join(frames_root, part_name)
    if not os.path.exists(part_frame_path):
        part_frame_path = part_frame_path.replace('/root/disk3/DFDC_images', '/root/disk4/deepfake/dataset/DFDC_images')
    for video_name in os.listdir(part_frame_path):
        json_name = video_name + '.json'
        video_path = join(part_frame_path, video_name)
        json_path = join(part_landmark_path, json_name)
        thread = threading.Thread(target=load_video_to_face, args=(video_path, json_path))
        threads.append(thread)
        thread.start()
        if len(threads) > 400:
            [thread.join() for thread in threads]
            threads = []

[thread.join() for thread in threads]
