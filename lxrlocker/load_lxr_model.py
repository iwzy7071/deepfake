import torch
from model.efficientnet.model import EfficientNet

net = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2})
model_dict = net.state_dict()

weight = {}
for key, value in torch.load('/root/data/wzy/lxrlocker/scale2models_lxr/PAI_tracking_45part_0.1resize_epoch_55_checkpoint.pth').items():
    key = key.replace('backbone.', '')
    if '_fc' in key:
        continue
    key = key.replace('last_linear', '_fc')
    weight[key] = value

model_dict.update(weight)
net.load_state_dict(model_dict)
torch.save(net.state_dict(), "/root/data/wzy/lxrlocker/scale2models_wzy/wzy_e_55_track_r01.pth")
