import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import cv2
import json
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
factors = 0.5


class PatchAllface(Dataset):
    def __init__(self, video_path, resize=299):
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        video = cv2.VideoCapture(video_path)
        video_name = video_path.split('/')[-1]
        video_name = video_name[:video_name.index('.')] + '.json'
        video_json_path = os.popen(
            'find {} -name {}'.format('/root/disk2/tracking_landmarks/DFDC_face_tracking', video_name))
        video_json_path = video_json_path.readlines()[0]
        video_json_path = video_json_path.strip('\n')
        anno = json.load(open(video_json_path))
        frame_num = 0

        image_names = []
        while True:
            ready, frame = video.read()
            if not ready:
                break
            for bbox in anno:
                left, top, right, bottom = int(bbox[frame_num][0]), int(bbox[frame_num][1]), \
                                           int(bbox[frame_num][2]), int(bbox[frame_num][3])
                height, width = frame.shape[:2]
                left = max(left, 0)
                right = min(width, right)
                bottom = min(height, bottom)
                top = max(0, top)
                face_width = right - left
                face_height = bottom - top

                left = int(max(left - face_width * factors, 0))
                right = int(min(width, right + face_width * factors))
                bottom = int(min(height, bottom + face_height * factors))
                top = int(max(0, top - face_height * factors))
                face = frame[top:bottom, left:right, :]
                image_names.append(face)
                if frame_num == 0:
                    cv2.rectangle(frame, (left + 1, top + 1), (right - 1, bottom - 1), (255, 0, 0), 2, 1)
                    ret = cv2.putText(frame, str(anno.index(bbox)),
                                      (int(0.5 * left + 0.5 * right), int(0.5 * top + 0.5 * bottom)),
                                      cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (100, 200, 200), 4)
                    cv2.imwrite('/root/data/wzy/_statistics/statistics/{}_first_frame.png'.format(video_name), ret)

            frame_num += 1

        self.image_names = image_names
        self.face_length = len(anno)

    def __getitem__(self, idx):
        image = self.image_names[idx]
        label = int(idx) % self.face_length
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)


from model.efficientnet.model import EfficientNet

net = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}).eval().cuda()
net.load_state_dict(torch.load('/root/data/wzy/checkpoint/efficient-video/14.pth'))


def get_statistic_graph(video_path):
    global net
    dataset = PatchAllface(video_path, resize=256)
    test_data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=300, num_workers=1)

    batch_idx = 0
    probs = {}
    with torch.no_grad():
        for patch_face, labels in test_data_loader:
            batch_idx += 1
            X = Variable(patch_face.cuda())
            print(X.size())
            y_pred = net(X)
            if type(y_pred) == tuple:
                y_pred = y_pred[0]
            y_pred = F.softmax(y_pred, dim=1)
            prob = y_pred[:, 1].cpu().numpy().tolist()
            labels = labels.numpy().tolist()
            for index in range(len(labels)):
                probs.setdefault(labels[index], []).append(prob[index])

    count = 0
    for prob in probs.keys():
        prob = probs[prob]
        plt.figure()
        indexes = range(len(prob))
        plt.bar(indexes, prob, width=0.8, color=['g'])
        video_name = video_path.split('/')[-1]
        plt.savefig(
            '/root/overface_statistics_graph/{}_performace_face_{}.png'.format(video_name, count))
        count += 1


from os.path import join

if __name__ == '__main__':
    for videoname in tqdm(os.listdir('/root/download_overface_video')):
        get_statistic_graph(join('/root/download_overface_video', videoname))
