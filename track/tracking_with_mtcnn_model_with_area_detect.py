import torch
import cv2
from tqdm import tqdm
import os
import random
from model.efficientnet.model import EfficientNet
from torch.nn import functional as F

random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from lxrlocker.kagglefile.kaggle_lxr import MyModel, EfficientNet as lxrEfficientNet

wzy_net = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}).cuda().eval()
wzy_net.load_state_dict(torch.load('/root/data/wzy/checkpoint/efficient-video/14.pth'))
lxr_net = lxrEfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2})
lxr_net = MyModel(lxr_net).cuda().eval()
lxr_net.load_state_dict(torch.load('/root/data/wzy/statistic/new_50partpai_tracking2_resize.pth'))

from facenet_pytorch import MTCNN
import numpy as np


class FastMTCNN(object):
    def __init__(self, *args, **kwargs):
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frame):
        boxes, scores = self.mtcnn.detect([frame])
        boxes, scores = boxes[0], scores[0]
        new_boxes = []
        for score_index, face_score in enumerate(scores):
            if face_score is None or face_score < 0.94:
                continue
            new_boxes += [boxes[score_index]]
        return new_boxes


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


video_trackers = [Tracker(0, 'SiamRPNotb', '/root/data/wzy/tracking_model/models/SiamRPNOTB.model') for _ in range(5)]

with torch.no_grad():
    for filename in tqdm(filenames):
        video_name = filename.split('/')[-1]
        if video_name not in name2path:
            continue
        label = 1 if name2path[video_name] == 'FAKE' else 0
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(7))
        trackers = []
        tracking_result = {}
        final_tracking_result = []
        current_tracking_epoch = 0
        tracking_counter = 0
        previous_box = {}
        for index in range(v_len):
            success = v_cap.grab()
            if not success:
                break

            if len(trackers) == 0:
                _, frame = v_cap.retrieve()
                first_boxes = fast_retinaface(frame)
                if len(tracking_result.keys()) > 0:
                    final_tracking_result.append(tracking_result)
                    tracking_result = {}


                if len(first_boxes) == 0:
                    tracking_counter = 2
                    continue

                for box_index, box in enumerate(first_boxes):
                    bbox = (box[0], box[1], box[2] - box[0], box[3] - box[1])
                    target_pos, target_sz = rect_2_cxy_wh(bbox)
                    video_trackers[box_index].init_tracker(frame, target_pos, target_sz)
                    trackers.append(box_index)
                    left, top, right, bottom = crop_box(box, frame)
                    tracking_result.setdefault(box_index, []).append(frame[top:bottom, left:right, :])
                    previous_box[box_index] = box

                continue

            if index % 2 != 0:
                continue

            _, frame = v_cap.retrieve()
            new_tracker_index = []
            for tracker_index in trackers:
                box, score = video_trackers[tracker_index].update_tracker(frame)

                if score < 0.5 or index % 28 == 0:
                    left, top, width, height = previous_box[tracker_index]
                    frame_height, frame_width = frame.shape[:2]
                    left, top = max(left - 2 * width, 0), max(top - 2 * height, 0)
                    right, bottom = min(left + 3 * width, frame_width), min(top + 3 * height, frame_height)
                    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                    box = fast_mtcnn(frame[top:bottom, left:right, :])

                    if len(box) != 1:
                        continue

                    box = box[0]
                    box = (box[0], box[1], box[2] - box[0], box[3] - box[1])
                    target_pos, target_sz = rect_2_cxy_wh(box)
                    video_trackers[tracker_index].init_tracker(frame[top:bottom, left:right, :], target_pos, target_sz)
                    new_tracker_index.append(tracker_index)
                    continue

                left, top, width, height = box
                previous_box[tracker_index] = box
                left, top, right, bottom = int(left), int(top), int(left + width), int(top + height)
                left, top, right, bottom = crop_box([left, top, right, bottom], frame)
                tracking_result.setdefault(box_index, []).append(frame[top:bottom, left:right, :])
                new_tracker_index.append(tracker_index)
            trackers = new_tracker_index

        v_cap.release()
        if len(tracking_result) < 10:
            trackers_indexes = np.linspace(0, len(tracking_result) - 1, 10, dtype=int)
            faces = [tracking_result[index] for index in trackers_indexes]
        else:
            faces = random.sample(tracking_result, 10)

        faces = [transform(Image.fromarray(face)) for face in faces]
        faces = torch.stack(faces, 0).unsqueeze(dim=0).cuda()
        lxr_pred = lxr_net(faces)
        lxr_pred = F.softmax(lxr_pred, dim=1)
        lxr_prob = lxr_pred.cpu().numpy().tolist()[0][1]

        wzy_pred = wzy_net(faces)
        wzy_pred = F.softmax(wzy_pred, dim=1)
        wzy_prob = wzy_pred.cpu().numpy().tolist()[0][1]

        online_performance.append([label, wzy_prob, lxr_prob])

json.dump(online_performance,
          open('/root/data/wzy/statistic/offline_tracking_with_global_area_detect_separate_tracks.json', 'w'))
