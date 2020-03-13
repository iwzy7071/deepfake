import torch
import cv2
import time
from tqdm import tqdm
import numpy as np
import glob
import os
from os.path import join
import random
from PIL import Image
from model.efficientnet.model import EfficientNet
from torch.nn import functional as F

'''
Prepare part
'''
random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

net = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}).cuda()
net.load_state_dict(torch.load('/root/data/wzy/checkpoint/efficientnetmeanlast/14.pth'))
net.eval()

'''
MTCNN PART
'''
from facenet_pytorch import MTCNN


class FastMTCNN(object):
    def __init__(self, *args, **kwargs):
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frame):
        boxes, _ = self.mtcnn.detect([frame])
        boxes = boxes[0]
        return boxes


fast_mtcnn = FastMTCNN(margin=14, factor=0.5, keep_all=True, device='cuda')

'''
tracking part
'''

import sys

sys.path.append('/root/data/wzy/tracking_model')
sys.path.append('/root/data/wzy/tracking_model/utils')
from tracking_model.tracker import Tracker
from tracking_model.utils.rect_utils import rect_2_cxy_wh

'''
Predict part
'''

import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
filenames = json.load(open('/root/data/wzy/datalocker/two-face-video.json'))
filenames = [
    filename.replace('/root/disk4/deepfake/dataset/DFDC_videos/DFDC_videos', '/root/data/DFDC_videos/DFDC_videos') for
    filename in filenames]
random.shuffle(filenames)
time_elapse = []
probs = []

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def crop_box(box, frame, factors=0.5):
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


with torch.no_grad():
    for filename in tqdm(filenames[:100]):
        start = time.time()

        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(7))

        indexes_with_linespace = [_ for _ in range(v_len)][::3]
        detectedindexes = np.linspace(0, len(indexes_with_linespace) - 1, 7, dtype=int).tolist()
        detectedindexes = [indexes_with_linespace[index] for index in detectedindexes]

        trackers = []
        tracking_result = {}
        count = 0
        for index in range(v_len):
            success = v_cap.grab()
            if not success:
                break

            # No Face Has been Detected
            if len(trackers) == 0:
                _, first_frame = v_cap.retrieve()
                first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                first_boxes = fast_mtcnn(first_frame)
                if len(first_boxes) == 0:
                    continue

                trackers = [Tracker(0, 'SiamRPNotb', '/root/data/wzy/tracking_model/models/SiamRPNOTB.model') for _ in
                            range(len(first_boxes))]
                for box_index, box in enumerate(first_boxes):
                    bbox = (box[0], box[1], box[2] - box[0], box[3] - box[1])
                    target_pos, target_sz = rect_2_cxy_wh(bbox)
                    trackers[box_index].init_tracker(first_frame, target_pos, target_sz)
                    left, top, right, bottom = crop_box(box, first_frame)
                    tracking_result.setdefault(box_index, []).append(first_frame[top:bottom, left:right, :])

            # Skip index not to be detected
            if index % 3 != 0:
                continue

            count += 1
            _, frame = v_cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for tracker_index, tracker in enumerate(trackers):
                box = trackers[tracker_index].update_tracker(frame)
                if index in detectedindexes:
                    left, top, width, height = box
                    left, top, right, bottom = int(left), int(top), int(left + width), int(top + height)
                    left, top, right, bottom = crop_box([left, top, right, bottom], frame)
                    tracking_result.setdefault(tracker_index, []).append(frame[top:bottom, left:right, :])
        v_cap.release()

        trackers_length = [len(value) for value in tracking_result.values()]
        # In case one detected face is casual error
        if len(trackers_length) > 1 and max(trackers_length) != min(trackers_length):
            tracking_face_array = [tracking_result[trackers_length.index(max(trackers_length))]]
        else:
            tracking_face_array = tracking_result.values()

        face_detected_array = []
        for faces in tracking_face_array:
            face_detected_array += [[transform(Image.fromarray(face)) for face in faces]]

        pred_scores = []
        count = 0
        for faces in face_detected_array:
            faces = torch.stack(faces, 0).unsqueeze(dim=0).cuda()
            y_pred = net(faces)
            y_pred = F.softmax(y_pred, dim=1)
            prob = y_pred[0][1].item()
            pred_scores.append(prob)

        max_prob = max(pred_scores)
        elapse = time.time() - start


