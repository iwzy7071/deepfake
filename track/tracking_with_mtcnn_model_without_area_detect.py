import torch
import cv2
from tqdm import tqdm
import os
import random
from model.efficientnet.model import EfficientNet
from torch.nn import functional as F

random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from lxrlocker.kagglefile.kaggle_lxr import MyModel, EfficientNet as lxrEfficientNet

wzy_net = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}).cuda().eval()
wzy_net.load_state_dict(torch.load('/root/data/wzy/checkpoint/efficient-video/14.pth'))
lxr_net = lxrEfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2})
lxr_net = MyModel(lxr_net).cuda().eval()
lxr_net.load_state_dict(torch.load('/root/data/wzy/statistic/new_50partpai_tracking2_resize.pth'))

import numpy as np
from wzyretinaface.detect import detect_face_from_frame, load_retinaface_model
from PIL import Image


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


fast_retinaface = Retinaface()

import sys

sys.path.append('/root/data/wzy/tracking_model')
sys.path.append('/root/data/wzy/tracking_model/utils')
from tracking_model.tracker import Tracker
from tracking_model.utils.rect_utils import rect_2_cxy_wh

import json
import glob

dataset = json.load(open('/root/data/wzy/statistic/statistics_real_fake_dataset.json'))
name2path = json.load(open('/root/data/wzy/datalocker/video_name2label.json'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
filenames = glob.glob('/root/disk4/deepfake/dataset/DFDC_videos/DFDC_videos/dfdc_train_part_15/*.mp4')
filenames = random.sample(filenames, 500)
time_elapse = []
probs = []
online_performance = []

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


online_performance = []
with torch.no_grad():
    for filename in tqdm(filenames):
        video_name = filename.split('/')[-1]
        if video_name not in name2path:
            continue
        label = 1 if name2path[video_name] == 'FAKE' else 0

        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(7))
        detectedindexes = np.linspace(10, v_len - 1, 7, dtype=int).tolist()
        trackers = []
        tracking_result = {}
        count = 0
        for index in range(v_len):
            success = v_cap.grab()
            if not success:
                break

            if len(trackers) == 0:
                _, first_frame = v_cap.retrieve()
                first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                first_boxes = fast_retinaface(first_frame)
                if len(first_boxes) == 0:
                    continue
                trackers = [Tracker(0, 'SiamRPNotb', '/root/data/wzy/tracking_model/models/SiamRPNOTB.model') for _
                            in range(len(first_boxes))]
                for box_index, box in enumerate(first_boxes):
                    bbox = (box[0], box[1], box[2] - box[0], box[3] - box[1])
                    target_pos, target_sz = rect_2_cxy_wh(bbox)
                    trackers[box_index].init_tracker(first_frame, target_pos, target_sz)
                    left, top, right, bottom = crop_box(box, first_frame)
                    tracking_result.setdefault(box_index, []).append(first_frame[top:bottom, left:right, :])

            # Skip index not to be detected
            if index not in detectedindexes:
                continue

            count += 1
            _, frame = v_cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for tracker_index, tracker in enumerate(trackers):
                box, score = trackers[tracker_index].update_tracker(frame)
                if score < 0.9:
                    continue
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

        wzy_pred_scores = []
        lxr_pred_scores = []
        count = 0
        for faces in face_detected_array:
            faces = torch.stack(faces, 0).unsqueeze(dim=0).cuda()
            wzy_pred = wzy_net(faces)
            wzy_pred = F.softmax(wzy_pred, dim=1)
            wzy_prob = wzy_pred[0][1].item()
            wzy_pred_scores.append(wzy_prob)

            lxr_pred = lxr_net(faces)
            lxr_pred = F.softmax(lxr_pred, dim=1)
            lxr_prob = lxr_pred[0][1].item()
            lxr_pred_scores.append(lxr_prob)

        wzy_avg_prob = sum(wzy_pred_scores) / len(wzy_pred_scores)
        lxr_avg_prob = sum(lxr_pred_scores) / len(lxr_pred_scores)

        online_performance.append([label, wzy_avg_prob, lxr_avg_prob])

json.dump(online_performance, open('/root/data/wzy/statistic/offline_tracking_with_global_detect.json', 'w'))
