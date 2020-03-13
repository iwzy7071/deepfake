import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from os.path import join

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
factors = 0.5


class PatchAllface(Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.dataset = json.load(open('/root/data/wzy/statistic/statistics_real_fake_dataset.json'))

    def __getitem__(self, idx):
        videopath, label = self.dataset[idx]
        image_names = sorted(os.listdir(videopath), key=lambda x: int(x.split('_')[0]))
        image_nums = np.linspace(0, len(image_names) - 1, 8, dtype=int, endpoint=True).tolist()
        image_names = [image_names[image_num] for image_num in image_nums]

        image_paths = [join(videopath, image_name) for image_name in image_names]
        images = [Image.open(imagepath).convert('RGB') for imagepath in image_paths]

        images = [self.transform(image) for image in images]
        image = torch.stack(images, dim=0)
        return image, label

    def __len__(self):
        return len(self.dataset)


from model.efficientnet.model import EfficientNet as VideoEfficientNet
from model.image_efficientnet.model import EfficientNet as ImageEfficientNet

image_net = ImageEfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}).eval().cuda()
image_net = nn.DataParallel(image_net).cuda()
image_net.load_state_dict(torch.load('/root/data/wzy/statistic/2.pth'))

video_net = VideoEfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}).eval().cuda()
video_net.load_state_dict(torch.load('/root/data/wzy/checkpoint/efficient-video/14.pth'))
video_net = nn.DataParallel(video_net).cuda()

dataset = PatchAllface()
test_dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=32, num_workers=1)

scores = {}
with torch.no_grad():
    for video, label in tqdm(test_dataloader):
        video, label = Variable(video.cuda()), Variable(label.cuda())
        video_pred = video_net(video)
        video_pred = F.softmax(video_pred, dim=1).cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        # video = video.view(video.size()[0] * 8, 3, 299, 299)
        # image_pred = image_net(video).view(-1, 8, 2)
        # image_pred = torch.mean(image_pred, dim=1)
        # image_pred = F.softmax(image_pred, dim=1).cpu().numpy().tolist()
        for vpred, lab in zip(video_pred, label):
            scores.setdefault('vpred', []).append([vpred[1], lab])
            # scores.setdefault('imgpred', []).append(imgpred[1])

json.dump(scores, open('value_statistics_distribution.json', 'w'))
