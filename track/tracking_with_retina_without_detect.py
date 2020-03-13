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

import numpy as np

from wzyretinaface.detect import detect_face_from_frame, load_retinaface_model
from PIL import Image


class Retinaface(object):
    def __init__(self):
        self.retina = load_retinaface_model('/root/data/wzy/resnet50.pth')

    def __call__(self, frames):
        faces = []
        for frame in frames:
            boxes, scores = detect_face_from_frame(self.retina, frame)
            for index in range(len(boxes)):
                box = boxes[index]
                score = scores[index]
                if score < 0.94:
                    continue
                left, top, right, bottom = crop_box(box, frame, factors=0.5)
                face = frame[top:bottom, left:right, :]
                faces.append(face)
        return faces


retinaface = Retinaface()

import sys

sys.path.append('/root/data/wzy/tracking_model')
sys.path.append('/root/data/wzy/tracking_model/utils')

import json

# dataset = json.load(open('/root/data/wzy/statistic/statistics_real_fake_dataset.json'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
filenames = json.load(open('/root/data/wzy/datalocker/three_kinds_video_dataset.json'))
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


online_performance = {}
for key, videos in filenames.items():
    for video in tqdm(videos):
        v_cap = cv2.VideoCapture(video)
        v_len = int(v_cap.get(7))
        detectedindexes = np.linspace(0, v_len - 1, 8, dtype=int).tolist()
        frames = []
        for index in range(v_len):
            success = v_cap.grab()
            if not success:
                break
            if index not in detectedindexes:
                continue
            _, frame = v_cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        v_cap.release()
        faces = retinaface(frames)
        faces = [Image.fromarray(face) for face in faces]
        faces = [transform(face) for face in faces]
        faces = torch.stack(faces, 0).unsqueeze(dim=0).cuda()
        lxr_pred = lxr_net(faces)
        lxr_pred = F.softmax(lxr_pred, dim=1)
        lxr_prob = lxr_pred.cpu().numpy().tolist()[0][1]

        wzy_pred = wzy_net(faces)
        wzy_pred = F.softmax(wzy_pred, dim=1)
        wzy_prob = wzy_pred.cpu().numpy().tolist()[0][1]
        online_performance.setdefault(key, []).append([wzy_prob, lxr_prob])

json.dump(online_performance, open('/root/data/wzy/statistic/different_people_numer_prob_with_global_detect.json', 'w'))
