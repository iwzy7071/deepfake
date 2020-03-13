import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import os
from sklearn.metrics import recall_score, accuracy_score
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from dataloader.dataloader_frame_299 import get_dataloader
from os.path import join
from lxrlocker.kagglefile.kaggle_lxr import EfficientNet, MyModel
from model.efficientnet.model import EfficientNet as wzyEfficient

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True

best_wP = 0
start_epoch = 5

print('==> Preparing data..')

BATCH_SIZE = 32
EPOCH = 20
INTERVAL = 200
model_name = 'efficient-all-frame-new'
save_path = join("/root/data/wzy/checkpoint", model_name)
train_dataloader, test_dataloader = get_dataloader(BATCH_SIZE)

print('==> Building model..')

net = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}).eval()
net = MyModel(net).cuda()
net.load_state_dict(torch.load('/root/data/wzy/lxrlocker/scale2models_lxr/45partmtcnnenhance.pth'))
wzy_net = wzyEfficient.from_name('efficientnet-b4', override_params={'num_classes': 2}).cuda()
# model_dict = wzy_net.state_dict()
#
# weight = {}
# for key, value in torch.load('/root/data/wzy/lxrlocker/scale2models_lxr/new_45partpai_tracking2_resize.pth').items():
#     key = key.replace('backbone.', '')
#     if '_fc' in key:
#         continue
#     key = key.replace('last_linear', '_fc')
#     weight[key] = value
#
# model_dict.update(weight)
wzy_net.load_state_dict(torch.load('/root/data/wzy/lxrlocker/scale2models_wzy/wzy_45partmtcnnenhance.pth'))


def test():
    net.eval()
    wzy_net.eval()
    batch_idx = 0
    with torch.no_grad():
        for video_images, video_label in tqdm(test_dataloader):
            batch_idx += 1
            video_images, video_label = Variable(video_images.cuda()), Variable(video_label.cuda())
            y_pred = net(video_images)
            y_pred = F.softmax(y_pred, dim=1)
            wzy_pred = wzy_net(video_images)
            wzy_pred = F.softmax(wzy_pred, dim=1)
            print(y_pred.sum() - wzy_pred.sum())
            print("lxr", y_pred[:, 1].cpu().detach().numpy())
            print("wzy", wzy_pred[:, 1].cpu().detach().numpy())
            exit()


test()
