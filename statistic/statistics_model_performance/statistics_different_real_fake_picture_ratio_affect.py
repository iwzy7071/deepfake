from PIL import Image
import json
import numpy as np
from torch.nn import functional as F

'''
get one pair real and fake video
'''

dataset = json.load(open('/root/data/wzy/datalocker/video_label_number_statistics.json'))['1']['FAKE']
fake2real = json.load(open('/root/data/wzy/datalocker/fake2real_vedio.json'))
name2path = json.load(open('/root/data/wzy/datalocker/video_name_to_path.json'))
video_paths = []
real_path = None
count = 0
for video_name in dataset:
    if video_name not in name2path:
        continue
    fake_path = name2path[video_name]
    if 'dfdc_train_part_9' in fake_path or 'dfdc_train_part_18' in fake_path or 'dfdc_train_part_28' in fake_path or 'dfdc_train_part_36' in fake_path or 'dfdc_train_part_48' in fake_path:
        real_name = fake2real[video_name]
        real_path = name2path[real_name]
        break

real_path = real_path.replace('/root/disk4/deepfake/dataset', '/root/data')
fake_path = fake_path.replace('/root/disk4/deepfake/dataset', '/root/data')


def crop_box(box, frame_size, factors=0.5):
    left, top, right, bottom = box
    width, height = frame_size
    left, right, bottom, top = max(left, 0), min(width, right), min(height, bottom), max(0, top)
    face_width = right - left
    face_height = bottom - top

    left = max(left - face_width * factors, 0)
    right = min(width, right + face_width * factors)
    bottom = min(height, bottom + face_height * factors)
    top = max(0, top - face_height * factors)

    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    box = np.array([left, top, right, bottom])

    return box


from facenet_pytorch import MTCNN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FastMTCNN(object):
    def __init__(self):
        self.mtcnn = MTCNN(device=device).eval()

    def __call__(self, allframes, stride=64):
        faces = []
        frames = []
        for frame in allframes:
            frames.append(frame)
            if len(frames) == 64 or frame == allframes[-1]:
                allboxes, allprobs = self.mtcnn.detect(frames, landmarks=False)
                for eachframe, eachbox, eachprob in zip(frames, allboxes, allprobs):
                    frame_size = eachframe.size
                    for index, prob in enumerate(eachprob):
                        if prob is None or prob < 0.93:
                            continue
                        box = eachbox[index]
                        box = crop_box(box, frame_size)
                        faces.append(eachframe.crop(box))
                frames = []
        return faces


fast_mtcnn = FastMTCNN()


def get_images_from_dict(image_dict):
    images = []
    for image_and_attribute in image_dict:
        image_path, attribute = image_and_attribute
        image = Image.open(image_path).convert('RGB')
        scores = attribute[1]
        boxes = attribute[0]
        for score_index, score in enumerate(scores):
            if score is None or score < 0.9:
                continue
            box = boxes[score_index]
            image = image.crop(crop_box(box, image.size))
            images.append(image)
    return images


import cv2
from tqdm import tqdm

with torch.no_grad():
    for fileindex, filename in enumerate([real_path, fake_path]):
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(7))
        act_vlen = 0
        for _ in range(v_len):
            success = v_cap.grab()
            if success:
                act_vlen += 1
            else:
                break
        v_cap.release()

        v_len = act_vlen
        v_cap = cv2.VideoCapture(filename)
        frames = []
        for index in tqdm(range(v_len)):
            success = v_cap.grab()
            if not success:
                break
            _, frame = v_cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        v_cap.release()

        if fileindex == 0:
            real_faces = fast_mtcnn(frames)
        else:
            fake_faces = fast_mtcnn(frames)

from lxrlocker.kagglefile.kaggle_lxr import MyModel, EfficientNet as lxrEfficientNet
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

import glob

pictures = glob.glob('/root/picture/*.png')
pictures = [Image.open(picture).convert('RGB') for picture in pictures]
pictures = [transform(face) for face in pictures]
real_faces = [transform(face) for face in real_faces]
fake_faces = [transform(face) for face in fake_faces]

lxr_net = lxrEfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2})
lxr_net = MyModel(lxr_net).cuda().eval()
lxr_net.load_state_dict(torch.load('/root/data/wzy/new_mtcnn_lxr.pth'))

online_perforance = {}
with torch.no_grad():
    for num_fake, num_real, num_picture in [[12, 0, 0], [9, 0, 3], [6, 0, 6], [3, 0, 9]]:
        fake_linespace_num = np.linspace(start=0, stop=len(fake_faces) - 1, num=num_fake, dtype=int)
        real_linespace_num = np.linspace(start=0, stop=len(real_faces) - 1, num=num_real, dtype=int)
        picture_linespace_num = np.linspace(start=0, stop=len(pictures) - 1, num=num_picture, dtype=int)
        faces = [real_faces[num] for num in real_linespace_num] + [fake_faces[num] for num in fake_linespace_num] + [
            pictures[num] for num in picture_linespace_num]
        faces = torch.stack(faces, dim=0).unsqueeze(dim=0).cuda()
        lxr_pred = lxr_net(faces)
        lxr_pred = F.softmax(lxr_pred, dim=1)
        lxr_prob = lxr_pred.cpu().numpy().tolist()[0][1]
        online_perforance['{}-{}-{}'.format(num_fake, num_real, num_picture)] = lxr_prob

print(online_perforance)
