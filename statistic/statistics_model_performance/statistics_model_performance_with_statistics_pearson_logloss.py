from math import log
from dataloader.dataloader_face_299 import get_dataloader
import torch
from tqdm import tqdm
import os
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def get_logloss(labels, preds):
    logloss = 0
    for label, prob in zip(labels, preds):
        logloss += label * log(prob) + log(1 - prob) * (1 - label)
    logloss = logloss / (-len(preds))
    return logloss


_, test_dataloader = get_dataloader(16)
from model.efficientnet.model import EfficientNet

models = [EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}),
          EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}),
          EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}),
          EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}),
          EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2})]

models[0].load_state_dict(torch.load('/root/data/wzy/lxrlocker/scale2models_wzy/mtcnn45_resume_mtcnn45_wzy.pth'))
models[1].load_state_dict(
    torch.load('/root/data/wzy/lxrlocker/scale2models_wzy/new_45partpai_tracking2_resize_wzy.pth'))
models[2].load_state_dict(torch.load('/root/data/wzy/lxrlocker/scale2models_wzy/PAI_dropblock-large_wzy.pth'))
models[3].load_state_dict(torch.load('/root/data/wzy/lxrlocker/scale2models_wzy/PAI_ori_retinaface_wzy.pth'))
models[4].load_state_dict(torch.load('/root/data/wzy/checkpoint/efficientnetmeanlast/14.pth'))

models[0] = nn.DataParallel(models[0]).cuda()
models[1] = nn.DataParallel(models[1]).cuda()
models[2] = nn.DataParallel(models[2]).cuda()
models[3] = nn.DataParallel(models[3]).cuda()
models[4] = nn.DataParallel(models[4]).cuda()

print(">>Finish Loading Efficient Model")
with torch.no_grad():
    online_result = {}
    true_labels = []
    for video_images, video_label in tqdm(test_dataloader):
        video_images, video_label = video_images.cuda(), video_label.cuda()
        for model_index, net in enumerate(models):
            y_pred = net(video_images)
            online_result.setdefault(model_index, [])
            online_result[model_index] += y_pred[:, 1].cpu().detach().numpy().tolist()

import json

json.dump(online_result, open('pearson_online_result.json', 'w'))
