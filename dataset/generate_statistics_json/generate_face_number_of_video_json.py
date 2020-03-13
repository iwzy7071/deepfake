import torch
import cv2
import time
from tqdm import tqdm
import numpy as np
import glob
import os
import random

random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
filenames = glob.glob('/root/disk4/deepfake/dataset/DFDC_videos/DFDC_videos/*/*.mp4')
random.shuffle(filenames)
time_elapse = []

from wzyretinaface.detect import load_retinaface_model, detect_face_from_frame

result = []
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

for filename in tqdm(filenames[:10000]):
    start = time.time()

    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(7))
    detectedindexes = np.linspace(0, v_len - 1, 8, dtype=int).tolist()

    first_boxes = []
    trackers = []
    tracking_result = []
    count = 0
    for index in range(v_len):
        success = v_cap.grab()
        if not success:
            break

        # No Face Has been Detected
        _, first_frame = v_cap.retrieve()
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_boxes = retinaface(first_frame)

        if len(first_boxes) == 0:
            continue

        if len(first_boxes) == 2:
            result.append(filename)
            break
        break

import json

json.dump(result, open('/root/data/wzy/datalocker/two-face-video.json', 'w'))
