import torch
import cv2
import time
from tqdm import tqdm
import numpy as np
import glob
import os
from os.path import join
import random
import sys

from trackers.tracker import Tracker
from trackers.utils.rect_utils import rect_2_cxy_wh, cxy_wh_2_rect, corner_to_rect, get_none_rotation_rect, \
    compute_success_overlap
import json

random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
filenames = json.load(open('/root/data/wzy/datalocker/two-face-video.json'))
random.shuffle(filenames)
time_elapse = []

from wzyretinaface.detect import load_retinaface_model, detect_face_from_frame

missing = 0
see_images = []
factor = 5


class Retinaface(object):
    def __init__(self):
        self.retina = load_retinaface_model('/root/data/wzy/resnet50.pth')

    def __call__(self, frame):
        new_boxes = []
        boxes, scores = detect_face_from_frame(self.retina, frame)
        for index in range(len(boxes)):
            box = boxes[index]
            score = scores[index]
            if score < 0.94:
                continue
            new_boxes.append([box])
        return new_boxes


retinaface = Retinaface()

for filename in tqdm(filenames[:300]):
    start = time.time()

    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(7))

    first_boxes = []
    trackers = []
    tracking_result = []
    count = 0
    for index in range(v_len):
        success = v_cap.grab()
        if not success:
            break

        # No Face Has been Detected
        if len(trackers) == 0:
            _, first_frame = v_cap.retrieve()
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            first_boxes = retinaface(first_frame)
            if len(first_boxes) == 0:
                continue
            trackers = [cv2.TrackerCSRT_create() for _ in range(len(first_boxes))]
            for box_index, box in enumerate(first_boxes):
                box = box[0]
                bbox = (box[0], box[1], box[2] - box[0], box[3] - box[1])
                trackers[box_index].init(first_frame, bbox)

        # Skip index not to be detected
        if index % factor != 0:
            continue
        count += 1
        _, frame = v_cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for tracker_index in range(len(trackers)):
            success, box = trackers[tracker_index].update(frame)
            if success:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                see_images.append(frame)

    v_cap.release()
    elapse = time.time() - start
    time_elapse.append(elapse)

average_time = sum(time_elapse) / len(time_elapse)
print(average_time)

dir_path = join('/root/tracking_image_two_people/', str(factor))
os.makedirs(dir_path, exist_ok=True)
for index, image in enumerate(random.sample(see_images, 100)):
    cv2.imwrite(join(dir_path, '{}.png'.format(index)), image)
