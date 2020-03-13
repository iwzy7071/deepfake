import torch
from lxrlocker.kagglefile.kaggle_lxr import EfficientNet, MyModel

net = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2})
net = MyModel(net)
weight = {}
for key, value in torch.load('/root/data/wzy/checkpoint/efficient-all-frame-new/6.pth').items():
    if '_fc' in key:
        fc_key = key.replace('_fc', 'last_linear')
        weight[fc_key] = value

    key = 'backbone.' + key
    print(key)
    weight[key] = value

# net.load_state_dict(weight)
# torch.save(net.state_dict(), "/root/data/wzy/lxrlocker/efficient-Acc0.875-All-Frame-45part-epoch-6.pth")
