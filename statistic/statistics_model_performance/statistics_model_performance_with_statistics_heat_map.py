from statistic.utils import *
from PIL import Image
import cv2
from os.path import join
from visualisation.core.utils import device
from visualisation.core.utils import image_net_postprocessing
from torchvision.transforms import ToTensor, Resize, Compose
from visualisation.core import *
from visualisation.core.utils import image_net_preprocessing
from matplotlib.animation import FuncAnimation
from collections import OrderedDict
import json
import numpy as np
import os
import torch

from lxrlocker.kagglefile.kaggle_lxr import MyModel, EfficientNet as lxrEfficientNet
from model.efficientnet.model import EfficientNet

wzy_net = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2}).cuda().eval()
wzy_net.load_state_dict(torch.load('/root/data/wzy/checkpoint/efficient-video/14.pth'))
lxr_net = lxrEfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2})
lxr_net = MyModel(lxr_net).cuda().eval()
lxr_net.load_state_dict(torch.load('/root/data/wzy/statistic/new_50partpai_tracking2_resize.pth'))

video_name2_path = json.load(open('/root/data/wzy/datalocker/video_name_to_path.json'))
video_name2_label = json.load(open('/root/data/wzy/datalocker/video_name2label.json'))

root_path = '/root/disk2/tracking_image_large_factor'
model_instances = [wzy_net, lxr_net]
model_names = ['wzy_net', 'lxr_net']


def get_frame_image(video_path):
    global video_name2_path
    video_path = video_name2_path[video_path]
    video_path = '/'.join(video_path.split('/')[-2:])
    video_path = join(root_path, video_path)
    video_path = video_path[:video_path.index('.')]

    image_names = os.listdir(video_path)
    image_num = np.linspace(0, len(image_names), 10, dtype=int, endpoint=False).tolist()
    image_names = [image_names[num] for num in image_num]
    image_names = [join(video_path, image_name) for image_name in image_names]
    image = [Image.open(image).convert('RGB') for image in image_names]
    return image


def get_heap_map(video_path):
    global model_names, model_instances, video_name2_label
    try:
        images = get_frame_image(video_path)
    except:
        return
    video_label = video_name2_label[video_path]
    if os.path.exists(join('/root/efficient_heat_map', '{}_{}.gif'.format(video_name, video_label))):
        return

    efficient_inputs = [Compose([Resize((299, 299)), ToTensor(), image_net_preprocessing])(x).unsqueeze(0) for x in
                        images]
    efficient_inputs = [x.unsqueeze(dim=0).cuda().requires_grad_(True) for x in efficient_inputs]

    model_outs = OrderedDict()

    images = list(map(lambda x: cv2.resize(np.array(x), (299, 299)), images))

    for name, model in zip(model_names, model_instances):
        module = model.cuda()
        module.eval()

        vis = GradCam(module, device)

        model_outs[name] = list(
            map(lambda x: tensor2img(vis(x, None, postprocessing=image_net_postprocessing)[0]), efficient_inputs))

        del module
        torch.cuda.empty_cache()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 20))
    axes = [ax2, ax3]

    def update(frame):
        all_ax = []
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.text(1, 1, 'Orig. Im', color="white", ha="left", va="top", fontsize=30)

        all_ax.append(ax1.imshow(images[frame]))
        for i, (ax, name) in enumerate(zip(axes, model_outs.keys())):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.text(1, 1, name, color="white", ha="left", va="top", fontsize=20)
            current_frame = (model_outs[name][frame] * 255).astype(np.uint8)
            all_ax.append(ax.imshow(current_frame, animated=True))
        return all_ax

    ani = FuncAnimation(fig, update, frames=range(len(images)), interval=1000, blit=True)
    fig.tight_layout()
    ani.save(join('/root/efficient_heat_map', '{}_{}.gif'.format(video_name, video_label)), writer='imagemagick')


if __name__ == '__main__':
    for video_name in os.listdir('/root/badcase_video/50part'):
        video_name = video_name.split('_')[0]
        get_heap_map(video_name)
