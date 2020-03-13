import sys
import os
import random

sys.path.append('/root/data/wzy/retinaface')
sys.path.append('/root/data/wzy/retina')
sys.path.append('/root/data/wzy/retina/rcnn')

from tqdm import tqdm
import numpy as np
from wzyretinaface.detect import detect_face_from_frame, load_retinaface_model
from retina.retinaface import RetinaFace
import glob
import cv2
from os.path import join

retina_detector = RetinaFace('/root/data/wzy/retina/models/R50-0000.params', 0, 0, 'net3')


class Retinaface(object):
    def __init__(self):
        self.retinaface = load_retinaface_model('/root/data/wzy/resnet50.pth')
        self.count = 0

    def __call__(self, frame):
        boxes, scores = detect_face_from_frame(self.retinaface, frame)
        new_boxes = []
        for score_index, face_score in enumerate(scores):
            if face_score is None or face_score < 0.94:
                continue
            new_boxes += [boxes[score_index]]
        return new_boxes


wzy_retinaface = Retinaface()


def lxr_retinaface(frame):
    thresh = 0.94
    scales = [1024, 1980]
    im_shape = frame.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
    flip = False
    detected_faces, _ = retina_detector.detect(frame, thresh, scales=scales, do_flip=flip)
    return detected_faces


count = 0
total = 0
video_paths = random.sample(glob.glob('/root/data/DFDC_videos/DFDC_videos/dfdc_train_part_4*/*.mp4'), 10000)
for video_path in tqdm(video_paths):
    v_cap = cv2.VideoCapture(video_path)
    v_len = int(v_cap.get(7))
    frames_nums = np.linspace(0, v_len - 1, 8, dtype=int).tolist()
    for index in range(v_len):
        success = v_cap.grab()
        if not success:
            break
        if index not in frames_nums:
            continue
        _, frame = v_cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lxr_faces = lxr_retinaface(frame)
        wzy_faces = wzy_retinaface(frame)
        total += 1
        if len(lxr_faces) == len(wzy_faces):
            continue

        if len(lxr_faces) > len(wzy_faces):
            for face_box in lxr_faces:
                p1 = (int(face_box[0]), int(face_box[1]))
                p2 = (int(face_box[2]), int(face_box[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.imwrite(join('/root/face_validate_second', 'lxr_more_face_detect_{}_{}.png'.format(count, total)),
                            frame)
                count += 1
        else:
            for face_box in wzy_faces:
                p1 = (int(face_box[0]), int(face_box[1]))
                p2 = (int(face_box[2]), int(face_box[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
                cv2.imwrite(join('/root/face_validate_second', 'wzy_more_face_detect_{}_{}.png'.format(count, total)),
                            frame)
                count += 1
