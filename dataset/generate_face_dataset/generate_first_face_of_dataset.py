from facenet_pytorch import MTCNN
import tqdm
import datetime
import smtplib
import os
import cv2
import numpy as np
import shutil
import torch
import glob
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
detector = MTCNN(device=torch.device('cuda'))


def detect_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final = []
    detected_faces_raw = detector.detect(img)
    if detected_faces_raw == []:
        return []
    confidences = []
    for n in detected_faces_raw:
        x, y, w, h = n['box']
        final.append([x, y, w, h])
        confidences.append(n['confidence'])
    if max(confidences) < 0.7:
        return []
    max_conf_coord = final[confidences.index(max(confidences))]
    return max_conf_coord


def crop(img, x, y, w, h):
    x -= 40
    y -= 40
    w += 80
    h += 80
    if x < 0:
        x = 0
    if y <= 0:
        y = 0
    return cv2.cvtColor(cv2.resize(img[y:y + h, x:x + w], (256, 256)), cv2.COLOR_BGR2RGB)


def detect_video(video):
    v_cap = cv2.VideoCapture(video)
    v_cap.set(1, NUM_FRAME)
    success, vframe = v_cap.read()
    vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
    bounding_box = detect_face(vframe)
    if bounding_box == []:
        count = 0
        current = NUM_FRAME
        while bounding_box == [] and count < MAX_SKIP:
            current += 1
            v_cap.set(1, current)
            success, vframe = v_cap.read()
            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            bounding_box = detect_face(vframe)
            count += 1
        if bounding_box == []:
            print('hi')
            return None
    x, y, w, h = bounding_box
    v_cap.release()
    return crop(vframe, x, y, w, h)


test_video_files = glob.glob('/root/disk4/deepfake/dataset/DFDC_videos/DFDC_videos/*/*.mp4')
MAX_SKIP = 10
NUM_FRAME = 150

for video_path in tqdm(test_video_files):
    img_file = detect_video(video_path)
    print(video)
    exit()
    cv2.imwrite('./DeepFake' + d_num + '/' + video.replace('.mp4', '').replace(test_dir, '') + '.jpg', img_file)
