#! pip install /kaggle/input/facenet-pytorch-wheel/facenet_pytorch-1.0.1-py3-none-any.whl
import sys

sys.path.append('/kaggle/input/')
retinafacePath = '/kaggle/input/retinaface/RetinaFace'
sys.path.append(retinafacePath)
import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch.utils.data.distributed
import models
from PIL import Image
import cv2
from facenet_pytorch import MTCNN
import random
import pandas as pd
import os
# sys.path.append('./RetinaFace')

from wzyretinaface import RetinaFace

from concurrent.futures import ThreadPoolExecutor

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

params = { \
    "crop_size": 299, \
    "val_size": 299, \
    "num_classes": 2, \
    "arch": "efficientnet_b4", \
    "resume_path": "/kaggle/input/ef4models/ef_b4_video_lr1e4_adam_32worker_30epochs_new_model_best.pth.tar", \
    "nb_frames": 10, \
    "detector": "RetinaFace", \
    "scale": 2, \
    "video": True, \
    "clip": True, \
    "normalize": True, \
    "debug": True, \
    "addition_params": {"thresh": 0.93, "scales": [1024, 1980]}}

random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)

transform = transforms.Compose([
    transforms.Resize((params["crop_size"], params["crop_size"])),
    transforms.CenterCrop(params["val_size"]),
    transforms.ToTensor(),
])

print("=> creating model '{}'".format(params["arch"]))


class VideoModel(nn.Module):
    def __init__(self, model, num_classes=2, embedding_size=1792):
        super(VideoModel, self).__init__()
        self.embedding_size = embedding_size
        self.backbone = model
        self.last_linear = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        B, N, C, H, W = x.size()
        frames = x.reshape(B * N, C, H, W)
        feature = self.backbone.forward(frames, feature=True)
        fusion_feature = torch.mean(feature.reshape(B, N, -1), dim=1)
        if params["normalize"]:
            fusion_feature = F.normalize(fusion_feature)
        out = self.last_linear(fusion_feature)
        return out


class FrameModel(nn.Module):
    def __init__(self, model, num_classes=2, embedding_size=1792):
        super(FrameModel, self).__init__()
        self.model = model

    def forward(self, x):
        feature = self.model.forward(x, feature=True)
        if params["normalize"]:
            feature = F.normalize(feature)
        out = self.model._dropout(feature)
        out = self.model._fc(out)
        return out


model = models.__dict__[params["arch"]](num_classes=params["num_classes"])

if params["video"]:
    model = VideoModel(model, num_classes=params["num_classes"])
else:
    model = FrameModel(model, num_classes=params["num_classes"])

if params["resume_path"]:
    # Use a local scope to avoid dangling references
    def resume():
        global best_prec1
        print("=> loading checkpoint '{}'".format(params["resume_path"]))
        if device == torch.device('cpu'):
            checkpoint = torch.load(params["resume_path"], map_location='cpu')
        else:
            checkpoint = torch.load(params["resume_path"])

        model_static = {}
        for key in checkpoint['state_dict'].keys():
            if params["video"]:
                model_static[key.replace('module.', '', 1)] = checkpoint['state_dict'][key]
            else:
                model_static[key.replace('module.', 'model.', 1)] = checkpoint['state_dict'][key]
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(model_static)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(params["resume_path"], checkpoint['epoch']))


    resume()
# print(model)
model = torch.nn.Sequential(model, torch.nn.Softmax(1))
model.to(device).eval()

if params["detector"] == "MTCNN":
    detector = MTCNN(device=device).eval()
else:
    # ctx = 0 mean gpu 0 ctx = -1 means cpu
    if device == torch.device('cpu'):
        detector = RetinaFace(retinafacePath + '/models/R50', 0, ctx_id=-1, network='net3')
    else:
        detector = RetinaFace(retinafacePath + '/models/R50', 0, ctx_id=0, network='net3')


# mtcnn = MTCNN(device=device).eval()



def expand_with_scale(bbox, s):
    l, t, r, b = bbox
    w = r - l
    h = b - t
    c_x = w / 2 + l
    c_y = h / 2 + t
    left = c_x - w * s / 2
    right = c_x + w * s / 2
    top = c_y - h * s / 2
    bottom = c_y + h * s / 2
    return left, right, top, bottom


def extract_and_preprocess(img, bbox, prob, transform, th=0.95, max_prob=True):
    img = np.array(img)
    faces = []
    max_conf = 0
    idx = 0
    if max_prob:
        for j in range(len(bbox)):
            if prob[j] > th:
                if prob[j] > max_conf:
                    max_conf = prob[j]
                    idx = j
        if max_conf > th:
            new_left, new_right, new_top, new_bottom = expand_with_scale(bbox[idx], params["scale"])
            h, w, c = img.shape
            face = img[max(0, int(new_top)):min(h, int(new_bottom)), max(0, int(new_left)):min(w, int(new_right)), :]
            faces.append(transform(Image.fromarray(face)) * 255.0)
    else:
        for j in range(len(bbox)):
            if prob[j] > th:
                new_left, new_right, new_top, new_bottom = expand_with_scale(bbox[j], params["scale"])
                h, w, c = img.shape
                face = img[max(0, int(new_top)):min(h, int(new_bottom)), max(0, int(new_left)):min(w, int(new_right)),
                       :]
                faces.append(transform(Image.fromarray(face)) * 255.0)

    return faces


def extract_faces_mtcnn(video_path, transform, max_prob=True):
    faces = []
    cap = cv2.VideoCapture(video_path)
    v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_len = 0
    for frame_num in range(v_len):
        succ = cap.grab()
        if succ:
            actual_len += 1
        else:
            break
    cap.release()
    if actual_len == 0:
        return faces
    sample = np.linspace(0, actual_len - 1, params["nb_frames"]).astype(int)
    imgs = []
    cap = cv2.VideoCapture(video_path)
    for i in range(actual_len):
        succ, frame = cap.read()
        if i in sample and succ:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(Image.fromarray(frame))
        if len(imgs) == params["nb_frames"]:
            break
    cap.release()
    bboxs, probs = detector.detect(imgs, landmarks=False)
    for i in range(len(bboxs)):
        if bboxs[i] is not None:
            faces_i = extract_and_preprocess(imgs[i], bboxs[i], probs[i], transform, max_prob=max_prob)
            faces += faces_i
    if len(faces) > 0:
        faces = torch.stack(faces)

    return faces


def extract_faces_retinaface(video_path, transform, max_prob=False):
    sample = []
    faces = []
    imgs = []

    cap = cv2.VideoCapture(video_path)
    v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = 0
    for j in range(v_len):
        success = cap.grab()
        if success:
            video_length += 1
        else:
            break
    cap.release()

    if video_length == 0:
        return faces

    sample = np.linspace(0, video_length - 1, params["nb_frames"]).astype(int)
    imgs = []
    cap = cv2.VideoCapture(video_path)
    for j in range(video_length):
        succ, image = cap.read()
        if j in sample and succ:
            imgs.append(image)
        if len(imgs) == params["nb_frames"]:
            break
    cap.release()
    thresh = 0.93
    scales = [1024, 1980]
    im_shape = imgs[0].shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.0
    # if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
    flip = False
    for im in imgs:
        bbox, _ = detector.detect(im, thresh, scales=scales, do_flip=flip)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        probs = []
        bboxes = []
        if bbox is not None:
            for i, bbox_with_score in enumerate(bbox):
                probs.append(bbox_with_score[4])
                bboxes.append(bbox_with_score[0:4])
            faces_i = extract_and_preprocess(Image.fromarray(im), bboxes, probs, transform, th=thresh,
                                             max_prob=max_prob)
            faces = faces + faces_i
    if len(faces) > 0:
        faces = torch.stack(faces)

    return faces


def process(video_path, transform):
    try:
        if params["detector"] == "MTCNN":
            faces = extract_faces_mtcnn(video_path, transform, max_prob=True)
        else:
            faces = extract_faces_retinaface(video_path, transform, max_prob=False)
        print("{} extract {} faces finished".format(video_path, len(faces)))
        print('face : {}'.format(faces.shape))
        if len(faces) == 0:
            pred = 0.5
        else:
            faces.sub_(mean).div_(std)
            if params["video"]:
                output = model(faces.unsqueeze(0).to(device))
            else:
                output = model(faces.to(device))
            score = output[:, 1].mean().item()
            pred = score
            if params["clip"]:
                pred = max(min(pred, 0.95), 0.05)
            print('output: ', video_path, pred)
        return pred
    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))
    return 0.5


def wokers(videos_list, num_workers):
    def process_single(i):
        y_pred = process(videos_list[i], transform)
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_single, range(len(videos_list)))

    return list(predictions)


video_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"
test_videos = sorted([vi for vi in os.listdir(video_dir) if vi[-4:] == ".mp4"])
if params["debug"]:
    test_videos = ['bnuwxhfahw.mp4']
test_videos_paths = []
for video_name in test_videos:
    video_path = os.path.join(video_dir, video_name)
    test_videos_paths.append(video_path)

predictions = wokers(test_videos_paths, num_workers=2)

submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
submission_df.to_csv("submission.csv", index=False)
